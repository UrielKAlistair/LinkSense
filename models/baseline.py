#!/usr/bin/env python3
"""Gradient-boosted and random-forest regressors over the choice set.

Each row is scored independently and the argmax within a group is the
chosen AP. That is a pointwise ranking approach: simple, and a sensible
default for a few hundred groups of tabular features, where boosting is
hard to beat. models/ranker.py adds a listwise alternative that optimises
the comparison directly.

Run:  python -m models.baseline data/dataset.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance

from .data import (LABEL_COL, assert_no_leakage, feature_columns, impute_features,
                   load_dataset, split_by_group, to_xy)
from .evaluate import evaluate_all, oracle_ceiling


def build_models(seed: int = 0) -> dict:
    return {
        "hist_gbr": HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, min_samples_leaf=8,
            l2_regularization=1.0, early_stopping=False, random_state=seed),
        "random_forest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=seed),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-target", action="store_true",
                        help="fit on log1p(throughput); throughput is heavily "
                             "right-skewed with a spike at zero, so squared "
                             "error on the raw scale is dominated by the "
                             "few large values")
    args = parser.parse_args()

    df = impute_features(load_dataset(args.dataset))
    feats = feature_columns(df)
    assert_no_leakage(feats)
    train, val, test = split_by_group(df, seed=args.seed)

    X_tr, y_tr = to_xy(train, feats)
    X_va, y_va = to_xy(val, feats)

    print(f"dataset={args.dataset}  rows={len(df)}  features={len(feats)}")
    print(f"groups: train={train.group_id.nunique()} val={val.group_id.nunique()} "
          f"test={test.group_id.nunique()}")
    oc = oracle_ceiling(val)
    print(f"val oracle: best={oc['mean_best_mbps']:.1f} random={oc['mean_random_choice_mbps']:.1f} "
          f"spread={oc['mean_spread_mbps']:.1f} Mbps\n")

    preds = {}
    fitted = {}
    for name, model in build_models(args.seed).items():
        target = np.log1p(y_tr) if args.log_target else y_tr
        model.fit(X_tr, target)
        p = model.predict(X_va)
        preds[name] = np.expm1(p) if args.log_target else p
        fitted[name] = model

    print(evaluate_all(val, preds).to_string(index=False))

    best = min(preds, key=lambda n: evaluate_all(val, {n: preds[n]})
               .set_index("model").loc[n, "mean_regret_mbps"])
    imp = permutation_importance(fitted[best], X_va,
                                 np.log1p(y_va) if args.log_target else y_va,
                                 n_repeats=15, random_state=args.seed, n_jobs=-1)
    order = np.argsort(imp.importances_mean)[::-1][:12]
    print(f"\ntop permutation importances ({best}):")
    for i in order:
        print(f"  {feats[i]:<34}{imp.importances_mean[i]:>8.4f} +/- {imp.importances_std[i]:.4f}")


if __name__ == "__main__":
    main()
