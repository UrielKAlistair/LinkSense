#!/usr/bin/env python3
"""Would generating more deployments still make the models better?

Simulating a new deployment is expensive, so before paying for more of them it
is worth knowing whether the models have stopped improving on the ones already
there. This script answers that by training on deliberately reduced slices of
the data and watching what the reduction costs.

For each repeat it draws one split, holds the test partition fixed, and then
trains on 12.5%, 25%, 50% and finally 100% of that repeat's training
topologies. Subsets are nested and drawn separately within each AP count, so a
smaller slice is always contained in the larger one and no deployment size
disappears. A topology is taken whole: every scan recorded at that
deployment moves together, since they are near-copies of each other and
splitting them would leak.

Models and hyperparameters are fixed rather than tuned per slice, so the curve
measures what the extra data bought and not what a wider search bought. The
heuristics are reported at zero training data as a flat reference line.

A curve that is still falling at 100% says more deployments would help. One
that has flattened says the next simulation run buys little, and the money is
better spent elsewhere.

Run:
  .venv/bin/python3 scripts/train/learning_curve.py data/v3_dataset.csv \
      --out results_v3/learning_curve.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.data import (feature_columns, scan_sample_weights, impute_features,  # noqa: E402
                         load_dataset, split_by_topology, to_xy)
from models.evaluate import baseline_predictions, selection_metrics  # noqa: E402


def subset_topologies(train: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Nested, AP-count-stratified subset of physical training topologies.

    Every scan belonging to a selected topology comes along with it. The
    permutation depends only on `seed`, so calling this with a larger fraction
    and the same seed returns a superset; change that and the curve stops
    comparing like with like.
    """
    topology = train.groupby("topology_id", sort=True).first().reset_index()
    rng = np.random.default_rng(seed)
    selected = []
    for _, stratum in topology.groupby("gt_n_aps", sort=True):
        members = stratum.topology_id.to_numpy()
        members = members[rng.permutation(len(members))]
        count = max(1, int(round(len(members) * fraction)))
        selected.extend(members[:count])
    return train[train.topology_id.isin(selected)].reset_index(drop=True)


def fit_models(train: pd.DataFrame, test: pd.DataFrame,
               features: list[str], seed: int) -> dict[str, np.ndarray]:
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X_train, y_train = to_xy(train, features)
    X_test, _ = to_xy(test, features)
    weights = scan_sample_weights(train)
    specifications = {
        "ridge_linear": (
            make_pipeline(StandardScaler(), Ridge(alpha=100.0)), False),
        "hist_gbr": (
            HistGradientBoostingRegressor(
                max_iter=400, learning_rate=0.06, min_samples_leaf=8,
                l2_regularization=1.0, early_stopping=False, random_state=seed), True),
        "random_forest": (
            RandomForestRegressor(
                n_estimators=400, min_samples_leaf=2, max_features=1.0,
                n_jobs=-1, random_state=seed), True),
    }

    predictions = {}
    for name, (model, log_target) in specifications.items():
        target = np.log1p(y_train) if log_target else y_train
        fit_args = ({"ridge__sample_weight": weights} if name == "ridge_linear" else
                    {"sample_weight": weights})
        model.fit(X_train, target, **fit_args)
        pred = model.predict(X_test)
        predictions[name] = np.expm1(np.clip(pred, -5, 12)) if log_target else pred
    return predictions


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/learning_curve.csv"))
    parser.add_argument("--fractions", type=float, nargs="+",
                        default=[0.125, 0.25, 0.5, 1.0],
                        help="shares of the training topologies to train on")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if any(not 0 < value <= 1 for value in args.fractions):
        parser.error("--fractions must lie in (0, 1]")
    args.fractions = sorted(set(args.fractions))
    return args


def main() -> int:
    args = parse_args()
    fractions = args.fractions

    frame = impute_features(load_dataset(args.dataset))
    features = feature_columns(frame)
    rows = []
    for repeat in range(args.repeats):
        full_train, _, test = split_by_topology(frame, seed=repeat)
        for name, prediction in baseline_predictions(test, fit_frame=full_train).items():
            if name == "random":
                continue
            rows.append({
                "split_seed": repeat,
                "train_fraction": 0.0,
                "train_topologies": 0,
                "train_groups": 0,
                "model": name,
                **selection_metrics(test, prediction),
            })
        for fraction in fractions:
            train = subset_topologies(full_train, fraction, seed=repeat)
            predictions = fit_models(train, test, features, seed=repeat)
            for name, prediction in predictions.items():
                rows.append({
                    "split_seed": repeat,
                    "train_fraction": fraction,
                    "train_topologies": train.topology_id.nunique(),
                    "train_groups": train.scan_id.nunique(),
                    "model": name,
                    **selection_metrics(test, prediction),
                })
            print(f"split={repeat} fraction={fraction:g} "
                  f"topologies={train.topology_id.nunique()}")

    result = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    # Only the trained rows form the curve; the heuristics sit at fraction 0.
    learned = result[result.train_fraction > 0]
    summary = learned.groupby(["model", "train_fraction", "train_topologies"])[
        ["mean_regret_mbps", "topology_mean_regret_mbps"]].agg(["mean", "std"])
    print("\n=== learning curve: held-out regret (Mbps) ===")
    print(summary.round(3).to_string())

    # The last step of the curve is the one that answers the question, so it is
    # printed on its own. A single fraction has no step to report.
    if len(fractions) >= 2:
        previous, final = fractions[-2:]
        means = learned.groupby(["model", "train_fraction"])[
            "topology_mean_regret_mbps"].mean().unstack()
        delta = pd.DataFrame({
            "previous_fraction": previous,
            "final_fraction": final,
            "previous_regret": means[previous],
            "final_regret": means[final],
            "improvement_mbps": means[previous] - means[final],
        })
        delta["improvement_fraction"] = (
            delta.improvement_mbps / delta.previous_regret.replace(0, np.nan))
        print("\n=== final learning-curve step (topology-balanced) ===")
        print(delta.round(3).to_string())
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
