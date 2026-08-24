#!/usr/bin/env python3
"""Measure whether tabular AP-selection models are still data-limited.

The test partition is fixed within each repeat. Increasing fractions draw
stratified subsets from that repeat's training topologies, always keeping all
five ns-3 seed realizations of a selected topology. Models and hyperparameters
are fixed so improvement reflects added data rather than a larger tuning search.

Run:
  python scripts/learning_curve.py data/dataset.csv --out results/learning_curve.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.data import (feature_columns, group_sample_weights, impute_features,  # noqa: E402
                         load_dataset, split_by_group, to_xy)
from models.evaluate import baseline_predictions, selection_metrics  # noqa: E402


def subset_topologies(train: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Nested, AP-count-stratified subset of physical training topologies."""
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
    weights = group_sample_weights(train)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/learning_curve.csv"))
    parser.add_argument("--fractions", default="0.125,0.25,0.5,1.0")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    fractions = [float(value) for value in args.fractions.split(",")]
    if not fractions or any(not 0 < value <= 1 for value in fractions):
        parser.error("--fractions must be comma-separated values in (0, 1]")
    fractions = sorted(set(fractions))

    frame = impute_features(load_dataset(args.dataset))
    features = feature_columns(frame)
    rows = []
    for repeat in range(args.repeats):
        full_train, _, test = split_by_group(frame, seed=repeat)
        for name, prediction in baseline_predictions(test).items():
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
                    "train_groups": train.group_id.nunique(),
                    "model": name,
                    **selection_metrics(test, prediction),
                })
            print(f"split={repeat} fraction={fraction:g} "
                  f"topologies={train.topology_id.nunique()}")

    result = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    learned = result[result.model.isin(["ridge_linear", "hist_gbr", "random_forest"])]
    summary = learned.groupby(["model", "train_fraction", "train_topologies"])[
        ["mean_regret_mbps", "topology_mean_regret_mbps"]].agg(["mean", "std"])
    print("\n=== learning curve: held-out regret (Mbps) ===")
    print(summary.round(3).to_string())

    if len(fractions) >= 2:
        previous, final = fractions[-2:]
        means = learned.groupby(["model", "train_fraction"])[
            "topology_mean_regret_mbps"].mean().unstack()
        if previous in means and final in means:
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
