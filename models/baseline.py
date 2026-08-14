#!/usr/bin/env python3
"""Tabular baselines for predicting candidate throughput from pre-association features.

Two reference points, deliberately kept simple:
  - DummyRegressor (predict the training mean): the score any real model
    must beat to have learned anything at all.
  - HistGradientBoostingRegressor: strong default for small tabular data,
    and the model most likely to *stay* the best here - with a few hundred
    rows and ~20 features, gradient boosting is usually hard to beat with
    a neural net.
  - RandomForestRegressor: second tree baseline, mostly to check the
    boosting result isn't an artifact of one particular learner.

Run directly to fit and report metrics:
    python -m models.baseline data/dataset.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from .data import feature_columns, impute_features, load_dataset, split, to_xy


def evaluate(name: str, model, X_train, y_train, X_eval, y_eval) -> dict:
    model.fit(X_train, y_train)
    pred = model.predict(X_eval)
    return {
        "model": name,
        "mae": float(mean_absolute_error(y_eval, pred)),
        "rmse": float(np.sqrt(np.mean((y_eval - pred) ** 2))),
        "r2": float(r2_score(y_eval, pred)),
    }


def build_models(seed: int = 0) -> dict:
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "hist_gbr": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.08, max_depth=None,
            min_samples_leaf=5, l2_regularization=1.0, random_state=seed,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = impute_features(load_dataset(args.dataset))
    feats = feature_columns(df)
    train_df, val_df, test_df = split(df, seed=args.seed)

    X_train, y_train = to_xy(train_df, feats)
    X_val, y_val = to_xy(val_df, feats)

    print(f"dataset={args.dataset} rows={len(df)} features={len(feats)}")
    print(f"split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"label mean={y_train.mean():.3f} std={y_train.std():.3f} Mbps\n")

    results = []
    for name, model in build_models(args.seed).items():
        results.append(evaluate(name, model, X_train, y_train, X_val, y_val))

    print(f"{'model':<16}{'MAE':>10}{'RMSE':>10}{'R2':>10}")
    for r in results:
        print(f"{r['model']:<16}{r['mae']:>10.3f}{r['rmse']:>10.3f}{r['r2']:>10.3f}")

    # permutation importance on the best non-dummy model, as a check that
    # the model is keying on plausible features (RSSI, channel activity)
    # rather than something accidental
    from sklearn.inspection import permutation_importance

    best = max((r for r in results if r["model"] != "dummy_mean"), key=lambda r: r["r2"])
    model = build_models(args.seed)[best["model"]]
    model.fit(X_train, y_train)
    imp = permutation_importance(model, X_val, y_val, n_repeats=10,
                                 random_state=args.seed, n_jobs=-1)
    order = np.argsort(imp.importances_mean)[::-1][:10]
    print(f"\ntop permutation importances ({best['model']}):")
    for i in order:
        print(f"  {feats[i]:<32}{imp.importances_mean[i]:>8.4f} +/- {imp.importances_std[i]:.4f}")


if __name__ == "__main__":
    main()
