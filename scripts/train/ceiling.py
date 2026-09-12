#!/usr/bin/env python3
"""How much better could an AP-selection model get if it stopped guessing?

A model here only ever sees what a client's radio picked up before it joined
anything. It has to infer the situation - how far away each access point is,
how many stations are already on it, how much traffic they are offering - from
signal levels and channel occupancy. Some of that inference must be wrong, and
the errors it causes are indistinguishable, in the results table, from the
model simply being a poor model.

This script separates the two by cheating on purpose. It trains the same kind
of model twice on the same splits:

  observed              the feat_* feature table, exactly what the client can
                        measure. This is the setting every other experiment in
                        the project reports.
  observed_plus_truth   the same table with the simulator's own record of the
                        scenario bolted on - the true distances, the true
                        station counts, the true offered loads.

The second model is handed everything the first one had plus the answers it
was trying to infer, so whatever it cannot predict is not an observation
problem. The distance between the two is the observation gap: the accuracy
that is lost purely because the client has to work the scenario out rather
than be told it. That gap is the room a better encoder has to compete for.

Two things this does not claim. The bound is empirical, not mathematical: the
richer input set is strictly larger, but a learner given more columns is not
guaranteed to do better on finite data. And it bounds this model class on this
feature table, not the sequence models, which read the scan rather than a
summary of it.

Some ground-truth columns are held out of the second model. Seeds are arbitrary
identifiers rather than physics. Association delay and observed duration are
outcomes of the run rather than inputs to it, so a model given them is reading
the future. Columns that never vary are dropped automatically.

Run:
  .venv/bin/python3 scripts/train/ceiling.py data/v3_dataset.csv \
      --out-dir results_v3/ceiling
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.data import (LABEL_COL, feature_columns, scan_sample_weights,  # noqa: E402
                         impute_features, load_dataset, split_by_topology)
from models.evaluate import (regression_metrics, selection_metrics,  # noqa: E402
                             validation_selection_key)

# Ground-truth columns that describe something other than the scenario the
# simulator was asked to build. Constant columns need no entry here: they carry
# no information and scenario_columns() drops them on sight.
NOT_SCENARIO_CONFIGURATION = {
    "gt_rng_seed", "gt_topology_seed",        # arbitrary identifiers
    "gt_assoc_delay", "gt_observed_seconds",  # outcomes of the run, not inputs
}


def scenario_columns(df: pd.DataFrame) -> list[str]:
    """Ground-truth columns describing the scenario, that a client cannot see."""
    return [c for c in df.columns
            if c.startswith("gt_")
            and c not in NOT_SCENARIO_CONFIGURATION
            and df[c].nunique() > 1]


def encode_scenario(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Numeric columns pass through; text columns become integer codes.

    Codes are assigned over the whole corpus rather than per split, so a value
    that appears only in test still maps to something defined. These are
    configuration labels - a placement stratum, the set of hotspot APs - and
    not measurements, so their coding carries no information about the label.

    Returned as a frame whose columns are prefixed, so it can be joined onto
    the dataset and then split alongside it; building a separate matrix and
    re-deriving row positions afterwards is how rows get silently misaligned.
    """
    encoded = {}
    for c in cols:
        values = df[c]
        if pd.api.types.is_numeric_dtype(values):
            encoded[f"truth_{c}"] = values.to_numpy(dtype=np.float32)
        else:
            encoded[f"truth_{c}"] = pd.Categorical(values).codes.astype(np.float32)
    return pd.DataFrame(encoded, index=df.index)


def fit_best_on_validation(train, val, test, cols, seed):
    """Fit a grid on train, keep whichever member decides best on validation.

    Returns that member's test predictions, its family name, and its validation
    score. The winner is not refitted: the estimator objects in the grid are
    distinct, so the one selected is still fitted on the training split.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

    grid = [
        ("hist_gbr", HistGradientBoostingRegressor(
            max_iter=n, learning_rate=lr, min_samples_leaf=leaf,
            l2_regularization=1.0, early_stopping=False, random_state=seed))
        for n in (300, 600) for lr in (0.03, 0.06, 0.1) for leaf in (4, 8, 16)
    ] + [
        ("random_forest", RandomForestRegressor(
            n_estimators=400, min_samples_leaf=leaf, n_jobs=-1, random_state=seed))
        for leaf in (1, 2, 4)
    ]

    X_train = train[cols].to_numpy(dtype=np.float32)
    y_train = train[LABEL_COL].to_numpy(dtype=np.float32)
    weights = scan_sample_weights(train)
    X_val = val[cols].to_numpy(dtype=np.float32)

    best = ((np.inf, np.inf), None, None)
    for name, estimator in grid:
        estimator.fit(X_train, y_train, sample_weight=weights)
        key = validation_selection_key(val, estimator.predict(X_val))
        if key < best[0]:
            best = (key, estimator, name)
    key, estimator, name = best
    return estimator.predict(test[cols].to_numpy(dtype=np.float32)), name, key[0]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("results/ceiling"))
    parser.add_argument("--repeats", type=int, default=5,
                        help="independent topology splits to average over")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = impute_features(load_dataset(args.dataset))
    observed = feature_columns(df)
    truth = scenario_columns(df)
    # Join the encoded scenario on before splitting, so both input sets are
    # columns of one frame and every split carries its own rows with it.
    df = pd.concat([df, encode_scenario(df, truth)], axis=1)
    observed_plus_truth = observed + [f"truth_{c}" for c in truth]

    print(f"rows={len(df)} scans={df.scan_id.nunique()} "
          f"topologies={df.topology_id.nunique()}")
    print(f"observed inputs: {len(observed)} feat_* columns")
    print(f"scenario inputs added on top ({len(truth)}): {truth}\n")

    rows = []
    for seed in range(args.repeats):
        train, val, test = split_by_topology(df, seed=seed)
        y_test = test[LABEL_COL].to_numpy(dtype=np.float32)
        for tag, cols in (("observed", observed),
                          ("observed_plus_truth", observed_plus_truth)):
            pred, estimator, val_regret = fit_best_on_validation(
                train, val, test, cols, seed)
            row = {"split_seed": seed, "inputs": tag, "estimator": estimator,
                   "n_inputs": len(cols), "val_topology_regret_mbps": val_regret}
            row.update(regression_metrics(y_test, pred))
            row.update(selection_metrics(test, pred))
            rows.append(row)
            print(f"  seed={seed} {tag:20s} est={estimator:14s} "
                  f"r2={row['r2']:.3f} r2_log={row['r2_log']:.3f} "
                  f"top1={row['top1_accuracy']:.3f} "
                  f"regret={row['mean_regret_mbps']:.2f}")

    out = pd.DataFrame(rows)
    out.to_csv(args.out_dir / "ceiling_raw.csv", index=False)
    summary = out.groupby("inputs")[["r2", "r2_log", "top1_accuracy",
                                     "mean_regret_mbps", "topology_mean_regret_mbps",
                                     "mean_spearman"]].agg(["mean", "std"])
    summary.to_csv(args.out_dir / "ceiling.csv")
    print("\n=== observed vs observed-plus-truth (held-out topologies) ===")
    print(summary.to_string())

    means = out.groupby("inputs")["topology_mean_regret_mbps"].mean()
    gap = means["observed"] - means["observed_plus_truth"]
    print(f"\nobservation gap: {gap:.3f} Mbps of regret "
          f"({means['observed']:.3f} observed, "
          f"{means['observed_plus_truth']:.3f} told the truth as well)")

    (args.out_dir / "inputs.json").write_text(json.dumps({
        "observed_columns": observed,
        "scenario_columns": truth,
        "observation_gap_mbps": float(gap),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
