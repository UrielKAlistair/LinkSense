#!/usr/bin/env python3
"""Tabular models given exactly the observation the transformer sees.

Comparing a transformer against models built on summary features confounds two
things: the architecture, and the fact that the two are fed different data. This
script removes the second. It reads the same binned corpus the transformer
trains on, reduces each option to a feature vector, and fits ordinary tabular
models on identical topology splits. Any remaining gap is representation.

Three encodings, in increasing faithfulness to the raw sequence:

  mean   every per-bin measurement averaged over the window
  rich   mean, std, min, max, last value and linear slope of each measurement
  flat   the whole sequence flattened, order preserved and nothing aggregated

Run:
  python scripts/train/aggregate_baseline.py data/v3_temporal.npz \
      --out-dir results_v3/aggregate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.evaluate import (baseline_predictions, random_selection_metrics,  # noqa: E402
                             regression_metrics, selection_metrics)
from models.ranker import SetRanker, train as train_ranker  # noqa: E402
from models.temporal import MaskedStandardizer, TemporalCorpus  # noqa: E402
from scripts.train.train_temporal import _flat_frame, _flatten_scores  # noqa: E402


def build_inputs(corpus: TemporalCorpus, encoding: str) -> np.ndarray:
    """(groups, options, features) under one of the three encodings."""
    temporal = corpus.temporal
    n_scans, n_options, n_steps, n_features = temporal.shape
    valid = corpus.time_mask[:, None, :, None]          # (G,1,T,1)
    counts = corpus.time_mask.sum(axis=1)[:, None, None].astype(np.float32)

    if encoding == "flat":
        # Zero the padded tails so absent bins cannot masquerade as readings.
        return (temporal * valid).reshape(n_scans, n_options, n_steps * n_features)

    masked = np.where(valid, temporal, np.nan)
    mean = np.nanmean(masked, axis=2)
    if encoding == "mean":
        return np.nan_to_num(mean).astype(np.float32)

    std = np.nan_to_num(np.nanstd(masked, axis=2))
    lo = np.nan_to_num(np.nanmin(masked, axis=2))
    hi = np.nan_to_num(np.nanmax(masked, axis=2))
    last = np.stack([
        temporal[g, :, np.flatnonzero(corpus.time_mask[g])[-1], :]
        for g in range(n_scans)])
    # Least-squares slope per option per feature, computed on valid bins only.
    slope = np.zeros_like(mean)
    for g in range(n_scans):
        idx = np.flatnonzero(corpus.time_mask[g])
        t = (idx - idx.mean()).astype(np.float32)
        denom = float((t ** 2).sum()) or 1.0
        slope[g] = np.einsum("t,otf->of", t, temporal[g][:, idx, :]) / denom
    del counts
    return np.concatenate(
        [np.nan_to_num(mean), std, lo, hi, last, slope], axis=-1).astype(np.float32)



def add_contrast(X: np.ndarray, option_mask: np.ndarray) -> np.ndarray:
    """Append the cross-option comparison a set encoder computes for free.

    A tabular model scores each option in isolation: it never sees the rivals.
    The transformer's permutation-invariant stage does, so any gap between them
    may be cross-option comparison rather than anything temporal. Appending the
    peer mean and this option's deviation from it hands the tabular model the
    same context, isolating that term.
    """
    m = option_mask[:, :, None].astype(np.float32)
    n = m.sum(axis=1, keepdims=True)                      # valid options per group
    total = (X * m).sum(axis=1, keepdims=True)
    peer_mean = np.divide(total - X * m, np.maximum(n - 1.0, 1.0))
    return np.concatenate([X, peer_mean * m, (X - peer_mean) * m], axis=-1).astype(np.float32)


def fit_trees(X, y, mask, train_idx, val_idx, test_idx, val_frame, seed):
    """HistGradientBoosting on flattened per-option rows, selected on val regret."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    def rows(indices):
        return np.concatenate([X[i][mask[i]] for i in indices])

    def targets(indices):
        return np.concatenate([y[i][mask[i]] for i in indices])

    def weights(indices):
        raw = np.concatenate([np.full(int(mask[i].sum()), 1.0 / mask[i].sum())
                              for i in indices])
        return raw / raw.mean()

    X_tr, y_tr, w_tr = rows(train_idx), targets(train_idx), weights(train_idx)
    best = ((np.inf, np.inf), None)
    for log_target in (False, True):
        target = np.log1p(y_tr) if log_target else y_tr
        for lr in (0.03, 0.06, 0.1):
            for leaf in (4, 8, 16):
                est = HistGradientBoostingRegressor(
                    max_iter=400, learning_rate=lr, min_samples_leaf=leaf,
                    l2_regularization=1.0, early_stopping=False, random_state=seed)
                est.fit(X_tr, target, sample_weight=w_tr)
                pred = est.predict(rows(val_idx))
                if log_target:
                    pred = np.expm1(np.clip(pred, -5, 12))
                sm = selection_metrics(val_frame, pred)
                key = (sm.get("topology_mean_regret_mbps", sm["mean_regret_mbps"]),
                       -sm["mean_spearman"])
                if key < best[0]:
                    best = (key, (lr, leaf, log_target))
    lr, leaf, log_target = best[1]
    est = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=lr, min_samples_leaf=leaf,
        l2_regularization=1.0, early_stopping=False, random_state=seed)
    est.fit(X_tr, np.log1p(y_tr) if log_target else y_tr, sample_weight=w_tr)
    pred = est.predict(rows(test_idx))
    return (np.expm1(np.clip(pred, -5, 12)) if log_target else pred)


def fit_mlp(X, y, mask, train_idx, val_idx, test_idx, seed, epochs, patience):
    """The same SetRanker used on feat_*, now fed the continuous observation."""
    def pack(indices):
        return (torch.from_numpy(X[indices]), torch.from_numpy(y[indices]),
                torch.from_numpy(mask[indices]),
                torch.from_numpy(np.zeros(mask[indices].shape, dtype=np.int64)))

    model = SetRanker(X.shape[-1], dropout=0.1)
    model = train_ranker(model, pack(train_idx), pack(val_idx), epochs, 3e-3, 1e-3,
                         alpha=0.7, temperature=5.0, seed=seed, verbose=False,
                         batch_size=32, patience=patience)
    model.eval()
    with torch.no_grad():
        scores = model(torch.from_numpy(X[test_idx]),
                       torch.from_numpy(mask[test_idx])).numpy()
    return np.expm1(np.clip(scores, -5, 10))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("results/aggregate"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--encodings", nargs="+", default=["mean", "rich", "flat"])
    parser.add_argument("--contrast", action="store_true",
                        help="append peer-mean and deviation-from-peer-mean features")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    corpus = TemporalCorpus.load(args.corpus)
    print(f"scans={len(corpus.scan_ids)} shape={corpus.temporal.shape}")
    encoded = {name: build_inputs(corpus, name) for name in args.encodings}
    if args.contrast:
        encoded = {k: add_contrast(v, corpus.option_mask) for k, v in encoded.items()}
    for name, X in encoded.items():
        print(f"  {name:5s} -> {X.shape[-1]} features per option")

    rows = []
    for seed in range(args.repeats):
        train_idx, val_idx, test_idx = corpus.split(seed=seed)
        test_frame = _flat_frame(corpus, test_idx)
        val_frame = _flat_frame(corpus, val_idx)
        y_test = test_frame["label_throughput_mbps"].to_numpy()
        print(f"\n--- split {seed}: train={len(train_idx)} test={len(test_idx)} ---")

        for name, X_raw in encoded.items():
            scaler = MaskedStandardizer(X_raw[train_idx], corpus.option_mask[train_idx])
            X = scaler(X_raw)
            for label, pred_full in (
                    ("mlp", fit_mlp(X, corpus.labels, corpus.option_mask,
                                    train_idx, val_idx, test_idx, seed,
                                    args.epochs, args.patience)),
            ):
                pred = _flatten_scores(corpus, test_idx, pred_full)
                row = {"split_seed": seed, "model": f"{label}_{name}"}
                row.update(regression_metrics(y_test, pred))
                row.update(selection_metrics(test_frame, pred))
                rows.append(row)
                print(f"    {label}_{name:5s} top1={row['top1_accuracy']:.3f} "
                      f"regret={row['mean_regret_mbps']:.3f} r2={row['r2']:.3f}")

            pred = fit_trees(X, corpus.labels, corpus.option_mask,
                             train_idx, val_idx, test_idx, val_frame, seed)
            row = {"split_seed": seed, "model": f"gbr_{name}"}
            row.update(regression_metrics(y_test, pred))
            row.update(selection_metrics(test_frame, pred))
            rows.append(row)
            print(f"    gbr_{name:5s} top1={row['top1_accuracy']:.3f} "
                  f"regret={row['mean_regret_mbps']:.3f} r2={row['r2']:.3f}")

        for name, pred in baseline_predictions(test_frame, _flat_frame(corpus, train_idx)).items():
            row = {"split_seed": seed, "model": name}
            row.update(random_selection_metrics(test_frame) if name == "random"
                       else selection_metrics(test_frame, pred))
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.out_dir / "results_raw.csv", index=False)
    summary = out.groupby("model")[["top1_accuracy", "mean_regret_mbps", "r2",
                                    "r2_log", "mean_spearman"]].agg(["mean", "std"])
    summary.to_csv(args.out_dir / "results.csv")
    print("\n=== tabular models on the CONTINUOUS observation ===")
    print(summary.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
