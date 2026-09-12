#!/usr/bin/env python3
"""A neural ranker that scores every AP in a scan at once.

The input is one scan: the summary features of each AP a client could
have joined, and the throughput each would have delivered. The output is one
score per AP, and the only thing asked of the scores is that the best AP gets
the highest one.

What distinguishes this from scoring each AP on its own is the middle step:

    per-option encoder  ->  h_i
    context             ->  mean and max of h over the options in the set
    score head          ->  s_i = f(h_i, context)

Because the context is a symmetric pool over the whole set, the score an
option gets depends on which rivals it faces but not on the order they are
listed in - a DeepSets-style permutation-invariant set encoder. That lets the
network say "this AP is only mildly loaded compared with the others here"
directly, rather than reading it off a precomputed feat_rel_* column.

Scans hold 2-8 discovered options and are padded to the largest in the
batch. Padding must never reach the pooling, the loss or the argmax; the tests
in tests/test_models.py check exactly that.

Training combines two objectives:
  ranking   a pairwise logistic loss over every pair of options in a set,
            weighted by how much throughput the pair actually differs by, so
            capacity goes to the mistakes that cost something.
  pointwise squared error against log1p(throughput), which keeps the scores
            readable as throughput estimates rather than bare rankings.

Run:  python -m models.ranker data/v3_dataset.csv --epochs 300
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import (SCAN_COL, LABEL_COL, assert_no_leakage, feature_columns,
                   impute_features, load_dataset, split_by_topology)
from .evaluate import TIE_TOL_MBPS, evaluate_all, label_spread


class Standardizer:
    """Centre and scale features using training rows only.

    Fitting on validation or test rows would let their distribution influence
    the inputs the model sees at training time, which is a quiet form of
    leakage. Features with no spread are left alone rather than divided by
    nearly zero.
    """

    def __init__(self, X: np.ndarray):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)


def pack_scans(df, feats: list[str], scaler: Standardizer | None = None):
    """Pack a frame into padded per-scan tensors (X, y, mask, row_index).

    row_index records where each option came from so predict_rows() can put
    the scores back in the frame's row order. It stores the frame's index
    values and uses them as positions, so `df` must carry a 0..n-1 index -
    which is what split_by_topology() returns. Passing a frame with any other
    index writes the scores to the wrong rows, or raises on a negative one.
    """
    if not df.index.equals(pd.RangeIndex(len(df))):
        raise ValueError(
            "pack_scans needs a frame indexed 0..n-1; call reset_index(drop=True) "
            "first, or the scores cannot be scattered back to their rows")
    scans = [g for _, g in df.groupby(SCAN_COL, sort=True)]
    max_n = max(len(g) for g in scans)

    X = np.zeros((len(scans), max_n, len(feats)), dtype=np.float32)
    y = np.zeros((len(scans), max_n), dtype=np.float32)
    mask = np.zeros((len(scans), max_n), dtype=bool)
    row_idx = np.full((len(scans), max_n), -1, dtype=np.int64)

    for i, g in enumerate(scans):
        n = len(g)
        raw = g[feats].to_numpy(dtype=np.float32)
        X[i, :n] = scaler(raw) if scaler is not None else raw
        y[i, :n] = g[LABEL_COL].to_numpy(dtype=np.float32)
        mask[i, :n] = True
        row_idx[i, :n] = g.index.to_numpy()
    return (torch.from_numpy(X), torch.from_numpy(y),
            torch.from_numpy(mask), torch.from_numpy(row_idx))


class SetRanker(nn.Module):
    """Scores each option, optionally conditioned on the whole option set.

    With use_context=False the network is a pointwise MLP of exactly the same
    shape and parameter count that never sees an option's rivals. That is the
    ablation scripts/train/ablation_context.py runs: a difference between the
    two modes can only come from the cross-option context.
    """

    def __init__(self, n_features: int, hidden: int = 96, embed: int = 64,
                 dropout: float = 0.1, use_context: bool = True):
        super().__init__()
        self.use_context = use_context
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, embed), nn.ReLU(),
        )
        # The head takes embed*3 inputs in both modes: the option's own
        # embedding plus two context slots. With use_context=False those slots
        # are fed zeros, so the two modes have identical parameter counts and
        # differ only in whether cross-option information reaches the head.
        self.head = nn.Sequential(
            nn.Linear(embed * 3, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)                       # (B, N, E)
        if not self.use_context:
            zeros = torch.zeros_like(h)
            return self.head(torch.cat([h, zeros, zeros], dim=-1)).squeeze(-1)
        m = mask.unsqueeze(-1).float()
        # masked pooling: padded slots must not influence the context
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        hmax = h.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        hmax = torch.nan_to_num(hmax, neginf=0.0)
        ctx = torch.cat([mean, hmax], dim=-1).unsqueeze(1).expand(-1, h.size(1), -1)
        return self.head(torch.cat([h, ctx], dim=-1)).squeeze(-1)


def ranking_loss(scores, y, mask, temperature: float = 5.0,
                 tie_tol: float = TIE_TOL_MBPS):
    """Pairwise ordering loss over the options in each scan.

    For every pair of real options in a set, the loss is small when the option
    with the higher throughput also has the higher score. Each pair is weighted
    by tanh(throughput gap / temperature), so a pair worth many Mbps counts for
    more than a near-tie, while the tanh stops one extreme pair from dominating
    the set. Pairs closer together than tie_tol are dropped entirely: ordering
    them correctly is worth nothing operationally.

    Only the score ORDER is constrained, never the score values, which is what
    lets pointwise_loss() pin the same scores to log1p(throughput) without the
    two objectives fighting.

    Sets with no scoring pair at all contribute nothing. The final term keeps a
    gradient path to `scores` when that is true of the whole batch, so backward
    does not fail on a batch of near-ties.
    """
    score_gap = scores.unsqueeze(2) - scores.unsqueeze(1)
    throughput_gap = y.unsqueeze(2) - y.unsqueeze(1)
    valid = mask.unsqueeze(2) & mask.unsqueeze(1)
    upper = torch.triu(torch.ones_like(valid, dtype=torch.bool), diagonal=1)
    valid &= upper & (throughput_gap.abs() > tie_tol)

    direction = throughput_gap.sign()
    weights = torch.tanh(throughput_gap.abs() / temperature).masked_fill(~valid, 0.0)
    losses = F.softplus(-direction * score_gap) * weights
    per_group = losses.sum(dim=(1, 2)) / weights.sum(dim=(1, 2)).clamp(min=1e-9)
    has_pairs = valid.any(dim=(1, 2))
    return per_group[has_pairs].mean() if has_pairs.any() else scores.sum() * 0.0


def pointwise_loss(scores, y, mask):
    diff = (scores - torch.log1p(y)) ** 2
    per_group = (diff.masked_fill(~mask, 0.0).sum(dim=1) /
                 mask.sum(dim=1).clamp(min=1))
    return per_group.mean()


def train(model, tr, va, epochs, lr, weight_decay, alpha, temperature, seed,
          verbose=True, batch_size=32, patience: int = 30):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xtr, ytr, mtr, _ = tr
    Xva, yva, mva, _ = va

    if epochs < 1:
        raise ValueError("train() needs at least one epoch to have any weights to return")
    best = (float("inf"), None)
    stale = 0
    n = Xtr.size(0)
    for epoch in range(epochs):
        model.train()
        # Minibatch over whole scans, never over rows: the ranking loss
        # is defined across the options within one set, so splitting a set
        # across two batches would silently drop the pairs that straddle them.
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            s = model(Xtr[idx], mtr[idx])
            loss = alpha * ranking_loss(s, ytr[idx], mtr[idx], temperature) + \
                (1 - alpha) * pointwise_loss(s, ytr[idx], mtr[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            sv = model(Xva, mva)
            vloss = (alpha * ranking_loss(sv, yva, mva, temperature) +
                     (1 - alpha) * pointwise_loss(sv, yva, mva)).item()
        if vloss < best[0]:
            best = (vloss, {k: v.detach().clone() for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if verbose and (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch + 1:>4}  train={loss.item():.4f}  val={vloss:.4f}  "
                  f"best={best[0]:.4f}")
        if stale >= patience:
            break

    # Return the best epoch's weights, not the last epoch's.
    model.load_state_dict(best[1])
    return model


def predict_rows(model, packed, n_rows: int) -> np.ndarray:
    """Scatter per-scan scores back into dataframe row order.

    `packed` is what pack_scans() returned for the frame being scored, and
    n_rows is that frame's length. Padded slots are dropped on the way out.
    """
    X, _, mask, row_idx = packed
    model.eval()
    with torch.no_grad():
        s = model(X, mask).numpy()
    out = np.zeros(n_rows, dtype=np.float64)
    idx = row_idx.numpy()
    m = mask.numpy()
    out[idx[m]] = s[m]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="weight on the ranking term (1-alpha on pointwise)")
    parser.add_argument("--temperature", type=float, default=5.0,
                        help="Mbps scale controlling pairwise regret weights")
    parser.add_argument("--patience", type=int, default=30,
                        help="stop after this many epochs without validation improvement")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = impute_features(load_dataset(args.dataset))
    feats = feature_columns(df)
    assert_no_leakage(feats)
    train_df, val_df, test_df = split_by_topology(df, seed=args.seed)

    scaler = Standardizer(train_df[feats].to_numpy(dtype=np.float32))
    tr = pack_scans(train_df, feats, scaler)
    va = pack_scans(val_df, feats, scaler)

    print(f"dataset={args.dataset} rows={len(df)} features={len(feats)}")
    print(f"scans: train={train_df.scan_id.nunique()} val={val_df.scan_id.nunique()} "
          f"test={test_df.scan_id.nunique()}")
    oc = label_spread(val_df)
    print(f"val oracle best={oc['mean_best_mbps']:.1f} random={oc['mean_random_choice_mbps']:.1f} Mbps\n")

    model = SetRanker(len(feats), dropout=args.dropout)
    model = train(model, tr, va, args.epochs, args.lr, args.weight_decay,
                  args.alpha, args.temperature, args.seed, patience=args.patience)

    pred = predict_rows(model, va, len(val_df))
    # scores are trained toward log1p(throughput); invert for reporting so
    # regression metrics are in Mbps like everything else
    print()
    print(evaluate_all(val_df, {"set_ranker": np.expm1(np.clip(pred, -5, 10))},
                       train_df).to_string(index=False))


if __name__ == "__main__":
    main()
