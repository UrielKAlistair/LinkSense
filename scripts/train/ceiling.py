#!/usr/bin/env python3
"""How much of the label is recoverable at all, and how much survives observation.

The ICC decomposition says ~98% of label variance is fixed by the static
scenario configuration. That bounds what a *time series* can add, but says
nothing about how hard that static configuration is to read off a passive
sniff. This script measures the two ends of that gap under one protocol:

  ceiling_gt      trained on gt_* ground truth - the true positions, station
                  counts and offered loads. A genie that knows the scenario
                  exactly. Its residual is irreducible simulator randomness.
  observed        trained on feat_* only - what the client can actually see.
                  This is the same model class as results/combined_tabular.

The difference between them is the observation gap: throughput accuracy lost
purely because the client must infer the scenario from frames rather than be
told it. That gap is the room any better *encoder* has to work in, which is
the question a set/sequence architecture is really competing for.

Excluded from the oracle on purpose:
  gt_rng_seed, gt_topology_seed   indices, not physics
  gt_assoc_delay, gt_observed_seconds   outcomes of the run, not inputs to it
  gt_ap_spacing, gt_packet_size   constant across the corpus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.data import (LABEL_COL, feature_columns, group_sample_weights,  # noqa: E402
                         impute_features, load_dataset, split_by_group, to_xy)
from models.evaluate import regression_metrics, selection_metrics  # noqa: E402

EXCLUDED_GT = {
    "gt_rng_seed", "gt_topology_seed",      # indices
    "gt_assoc_delay", "gt_observed_seconds",  # outcomes, not configuration
    "gt_ap_spacing", "gt_packet_size",      # constant
}


def oracle_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns
            if c.startswith("gt_") and c not in EXCLUDED_GT and df[c].nunique() > 1]
    return cols


def encode(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Numeric passthrough; categoricals become integer codes fitted corpus-wide.

    Category codes are derived from the whole corpus rather than the training
    split so that an unseen level in test maps to a defined value. These are
    configuration labels (a stratum name, a hotspot AP set), not measurements,
    so their coding carries no label information.
    """
    blocks = []
    for c in cols:
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            blocks.append(pd.Categorical(s).codes.astype(np.float32)[:, None])
        else:
            blocks.append(s.to_numpy(dtype=np.float32)[:, None])
    return np.concatenate(blocks, axis=1)


def fit_best(X_tr, y_tr, w_tr, X_va, X_te, val_df, seed):
    """Select on validation regret, exactly as scripts/train_eval.py does."""
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

    grid = []
    for n in (300, 600):
        for lr in (0.03, 0.06, 0.1):
            for leaf in (4, 8, 16):
                grid.append(("hist_gbr", HistGradientBoostingRegressor(
                    max_iter=n, learning_rate=lr, min_samples_leaf=leaf,
                    l2_regularization=1.0, early_stopping=False, random_state=seed)))
    for n in (400,):
        for leaf in (1, 2, 4):
            grid.append(("random_forest", RandomForestRegressor(
                n_estimators=n, min_samples_leaf=leaf, n_jobs=-1, random_state=seed)))

    best = ((np.inf, np.inf), None, None)
    for name, est in grid:
        est.fit(X_tr, y_tr, sample_weight=w_tr)
        pred = est.predict(X_va)
        sm = selection_metrics(val_df, pred)
        key = (sm.get("topology_mean_regret_mbps", sm["mean_regret_mbps"]),
               -sm["mean_spearman"])
        if key < best[0]:
            best = (key, est, name)
    return best[1].predict(X_te), best[2], best[0][0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("results/ceiling"))
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = impute_features(load_dataset(args.dataset))
    gt_cols = oracle_columns(df)
    feat_cols = feature_columns(df)
    print(f"rows={len(df)} groups={df.group_id.nunique()} "
          f"topologies={df.topology_id.nunique()}")
    print(f"oracle inputs ({len(gt_cols)}): {gt_cols}")
    print(f"observed inputs: {len(feat_cols)} feat_* columns\n")

    gt_all = encode(df, gt_cols)
    rows = []
    for seed in range(args.repeats):
        train, val, test = split_by_group(df, seed=seed)
        # split_by_group resets the index, so re-derive positions by group_id
        key = df.group_id.astype(str) + "|" + df.ap_index.astype(str)
        pos = pd.Series(np.arange(len(df)), index=key)

        def take(part):
            k = part.group_id.astype(str) + "|" + part.ap_index.astype(str)
            return pos.loc[k].to_numpy()

        i_tr, i_va, i_te = take(train), take(val), take(test)
        w_tr = group_sample_weights(train)
        y_tr = train[LABEL_COL].to_numpy(dtype=np.float32)
        y_te = test[LABEL_COL].to_numpy(dtype=np.float32)

        for tag, X in (("ceiling_gt", gt_all),
                       ("observed_feat", df[feat_cols].to_numpy(dtype=np.float32))):
            pred, chosen, val_regret = fit_best(
                X[i_tr], y_tr, w_tr, X[i_va], X[i_te], val, seed)
            row = {"split_seed": seed, "inputs": tag, "estimator": chosen,
                   "val_topology_regret_mbps": val_regret}
            row.update(regression_metrics(y_te, pred))
            row.update(selection_metrics(test, pred))
            rows.append(row)
            print(f"  seed={seed} {tag:14s} est={chosen:14s} "
                  f"r2={row['r2']:.3f} r2_log={row['r2_log']:.3f} "
                  f"top1={row['top1_accuracy']:.3f} "
                  f"regret={row['mean_regret_mbps']:.2f}")

    out = pd.DataFrame(rows)
    out.to_csv(args.out_dir / "ceiling_raw.csv", index=False)
    summary = out.groupby("inputs")[["r2", "r2_log", "top1_accuracy",
                                     "mean_regret_mbps", "mean_spearman"]].agg(["mean", "std"])
    summary.to_csv(args.out_dir / "ceiling.csv")
    print("\n=== ceiling vs observed (held-out topologies) ===")
    print(summary.to_string())
    (args.out_dir / "inputs.json").write_text(json.dumps(
        {"oracle_columns": gt_cols, "n_feat_columns": len(feat_cols)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
