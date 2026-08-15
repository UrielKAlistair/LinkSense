#!/usr/bin/env python3
"""A set-based listwise ranker over the APs available in one scenario.

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

Groups hold 2-4 options, so batches are padded to the largest group and
masked; padding must never leak into pooling, the loss, or the argmax.

Two objectives, combined by default:
  listwise  softmax cross-entropy between predicted scores and a softened
            distribution over the true throughputs. Optimises the ordering
            directly, which is what selection needs.
  pointwise MSE on log1p(throughput). Keeps the outputs interpretable as
            throughput estimates rather than bare scores, and regularises
            the ranking objective (which is indifferent to scale).

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
    def __init__(self, n_features: int, hidden: int = 96, embed: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, embed), nn.ReLU(),
        )
        # score head sees the option and the set it belongs to (mean + max)
        self.head = nn.Sequential(
            nn.Linear(embed * 3, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)                       # (B, N, E)
        m = mask.unsqueeze(-1).float()
        # masked pooling: padded slots must not influence the context
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        hmax = h.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        hmax = torch.nan_to_num(hmax, neginf=0.0)
        ctx = torch.cat([mean, hmax], dim=-1).unsqueeze(1).expand(-1, h.size(1), -1)
        return self.head(torch.cat([h, ctx], dim=-1)).squeeze(-1)


def listwise_loss(scores, y, mask, temperature: float = 5.0):
    """Softmax CE against a softened distribution over true throughputs.

    Temperature controls how sharply the target concentrates on the best
    option: too low and near-ties become arbitrary hard labels, too high and
    the target washes out to uniform.
    """
    neg = torch.finfo(scores.dtype).min
    log_p = torch.log_softmax(scores.masked_fill(~mask, neg), dim=1)
    target = torch.softmax((y / temperature).masked_fill(~mask, neg), dim=1)
    return -(target * log_p).masked_fill(~mask, 0.0).sum(dim=1).mean()


def pointwise_loss(scores, y, mask):
    diff = (scores - torch.log1p(y)) ** 2
    return diff.masked_fill(~mask, 0.0).sum() / mask.sum().clamp(min=1)


def train(model, tr, va, epochs, lr, weight_decay, alpha, temperature, seed, verbose=True):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xtr, ytr, mtr, _ = tr
    Xva, yva, mva, _ = va

    best = (float("inf"), None)
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        s = model(Xtr, mtr)
        loss = alpha * listwise_loss(s, ytr, mtr, temperature) + \
            (1 - alpha) * pointwise_loss(s, ytr, mtr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            sv = model(Xva, mva)
            vloss = (alpha * listwise_loss(sv, yva, mva, temperature) +
                     (1 - alpha) * pointwise_loss(sv, yva, mva)).item()
        if vloss < best[0]:
            best = (vloss, {k: v.detach().clone() for k, v in model.state_dict().items()})
        if verbose and (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch + 1:>4}  train={loss.item():.4f}  val={vloss:.4f}  best={best[0]:.4f}")

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
                        help="weight on the listwise term (1-alpha on pointwise)")
    parser.add_argument("--temperature", type=float, default=5.0)
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
                  args.weight_decay, args.alpha, args.temperature, args.seed)

    pred = predict_rows(model, va, len(val_df))
    # scores are trained toward log1p(throughput); invert for reporting so
    # regression metrics are in Mbps like everything else
    print()
    print(evaluate_all(val_df, {"set_ranker": np.expm1(np.clip(pred, -5, 10))}).to_string(index=False))
    if args.smoke_test:
        print("\nsmoke test OK: forward/backward/eval path runs end to end")


if __name__ == "__main__":
    main()
