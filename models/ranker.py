#!/usr/bin/env python3
"""A set-based pairwise ranker over the APs available in one scenario.

Why not just regress each option independently (models/baseline.py does
exactly that)? Because the quantity that matters is a comparison. The best
AP in a quiet deployment and the best AP in a congested one are chosen by
different reasoning, and a pointwise model has to rediscover the
alternatives from hand-built feat_rel_* columns. Here the model sees the
whole option set at once:

    per-option encoder  ->  h_i
    context             ->  pooled mean/max over the options in the group
    score head          ->  s_i = f(h_i, context)

which is a DeepSets-style permutation-invariant set encoder. The context
term lets the network express "this AP is only mildly loaded *compared to
the others here*" without that comparison being precomputed.

Groups hold 2-8 discovered options, so batches are padded to the largest group and
masked; padding must never leak into pooling, the loss, or the argmax.

Two objectives, combined by default:
  ranking   pairwise logistic loss over every valid AP pair, weighted by the
            true throughput gap. Optimises the ordering directly and spends
            little capacity separating operationally equivalent near-ties.
  pointwise MSE on log1p(throughput). Keeps the outputs interpretable as
            throughput estimates rather than bare scores. Pairwise ordering
            and log-throughput calibration are scale-compatible objectives.

Run:  python -m models.ranker data/dataset.csv --epochs 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import (GROUP_COL, LABEL_COL, assert_no_leakage, feature_columns,
                   impute_features, load_dataset, split_by_group)
from .evaluate import evaluate_all, oracle_ceiling


class Standardizer:
    """Fitted on training rows only; eval statistics must not leak in."""

    def __init__(self, X: np.ndarray):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)


def make_groups(df, feats: list[str], scaler: Standardizer | None = None):
    """Pack a frame into padded per-group tensors (X, y, mask, row_index)."""
    order = []
    groups = []
    for gid, g in df.groupby(GROUP_COL, sort=True):
        order.append(gid)
        groups.append(g)
    max_n = max(len(g) for g in groups)

    X = np.zeros((len(groups), max_n, len(feats)), dtype=np.float32)
    y = np.zeros((len(groups), max_n), dtype=np.float32)
    mask = np.zeros((len(groups), max_n), dtype=bool)
    row_idx = np.full((len(groups), max_n), -1, dtype=np.int64)

    for i, g in enumerate(groups):
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

    use_context=False turns this into a plain pointwise MLP of the same
    capacity, which is the ablation that isolates what the set context is
    actually worth: any difference between the two is attributable to
    seeing the alternatives rather than to depth or parameter count.
    """

    def __init__(self, n_features: int, hidden: int = 96, embed: int = 64,
                 dropout: float = 0.1, use_context: bool = True):
        super().__init__()
        self.use_context = use_context
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, embed), nn.ReLU(),
        )
        # The head input width is the same either way. Without context the
        # context slots are fed zeros, so the ablation carries strictly no
        # cross-option information while keeping the parameter count
        # identical - otherwise a win for the context model could just be a
        # win for having more parameters.
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


def ranking_loss(scores, y, mask, temperature: float = 5.0, tie_tol: float = 0.5):
    """Regret-weighted pairwise ordering loss.

    A softmax target formed from linear Mbps requires optimal scores to be
    linear in Mbps, while the calibration term below requires the same scores
    to equal log1p(Mbps). Those objectives cannot both be satisfied. Pairwise
    logistic loss requires only the correct score order, so it is compatible
    with calibrated log-throughput outputs.

    Throughput gaps at or below tie_tol carry no weight. Above it, tanh(gap /
    temperature) smoothly gives costly mistakes more influence without
    allowing one extreme pair to dominate a group.
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
          verbose=True, batch_size=32, patience: int | None = 30):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xtr, ytr, mtr, _ = tr
    Xva, yva, mva, _ = va

    best = (float("inf"), None)
    stale = 0
    n = Xtr.size(0)
    for epoch in range(epochs):
        model.train()
        # Minibatch over GROUPS (a group is the atomic unit - the ranking
        # loss is defined across the options within one). Full-batch descent
        # gave one step per epoch, far too few to fit this model.
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
        if patience is not None and stale >= patience:
            break

    if best[1] is not None:
        model.load_state_dict(best[1])
    return model


def predict_rows(model, packed, n_rows: int) -> np.ndarray:
    """Scatter per-group scores back to dataframe row order."""
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
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    df = impute_features(load_dataset(args.dataset))
    feats = feature_columns(df)
    assert_no_leakage(feats)
    train_df, val_df, test_df = split_by_group(df, seed=args.seed)

    scaler = Standardizer(train_df[feats].to_numpy(dtype=np.float32))
    tr = make_groups(train_df, feats, scaler)
    va = make_groups(val_df, feats, scaler)

    print(f"dataset={args.dataset} rows={len(df)} features={len(feats)}")
    print(f"groups: train={train_df.group_id.nunique()} val={val_df.group_id.nunique()} "
          f"test={test_df.group_id.nunique()}")
    oc = oracle_ceiling(val_df)
    print(f"val oracle best={oc['mean_best_mbps']:.1f} random={oc['mean_random_choice_mbps']:.1f} Mbps\n")

    model = SetRanker(len(feats), dropout=args.dropout)
    model = train(model, tr, va, 3 if args.smoke_test else args.epochs, args.lr,
                  args.weight_decay, args.alpha, args.temperature, args.seed,
                  patience=args.patience)

    pred = predict_rows(model, va, len(val_df))
    # scores are trained toward log1p(throughput); invert for reporting so
    # regression metrics are in Mbps like everything else
    print()
    print(evaluate_all(val_df, {"set_ranker": np.expm1(np.clip(pred, -5, 10))}).to_string(index=False))
    if args.smoke_test:
        print("\nsmoke test OK: forward/backward/eval path runs end to end")


if __name__ == "__main__":
    main()
