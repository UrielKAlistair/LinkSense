"""Shared dataset loading/splitting for the AP-selection throughput models.

Enforces the feat_/gt_/label_ column convention from
scripts/build_dataset.py: only feat_* columns are ever used as model
input, gt_* is for analysis/leakage-checking only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_COL = "label_throughput_mbps"
# sentinel for "this AP/rival was never heard during the pre-association
# window" - far below any real observed RSSI (roughly -30..-90 dBm in this
# sim), so it's distinguishable from a genuinely weak-but-heard signal
MISSING_RSSI_SENTINEL = -100.0


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("feat_")]


def ground_truth_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("gt_")]


def load_dataset(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = {LABEL_COL} - set(df.columns)
    if missing:
        raise ValueError(f"dataset missing required columns: {missing}")
    return df


def impute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaNs in feat_* columns that arise when an AP/rival wasn't heard.

    feat_target_seen / feat_rival_count already record *whether* that
    happened, so this only needs to give the RSSI/gap columns a
    well-defined, out-of-distribution value rather than a NaN - safe for
    models (like the MLP) that can't take NaN input directly. Tree models
    (HistGradientBoostingRegressor) would be fine with the raw NaNs, but
    using the same imputed frame for both keeps the comparison apples-to-
    apples.
    """
    df = df.copy()
    rssi_like = [c for c in feature_columns(df) if "rssi" in c]
    for col in rssi_like:
        df[col] = df[col].fillna(MISSING_RSSI_SENTINEL)
    other_feat = [c for c in feature_columns(df) if c not in rssi_like]
    for col in other_feat:
        df[col] = df[col].fillna(0.0)
    return df


def split(df: pd.DataFrame, val_frac: float = 0.2, test_frac: float = 0.2, seed: int = 0):
    """Random split - each row is an independent sampled scenario (see
    scripts/run_sweep.py), so no grouping/leakage concern across rows."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_val = int(len(df) * val_frac)
    n_test = int(len(df) * test_frac)
    val_idx = idx[:n_val]
    test_idx = idx[n_val:n_val + n_test]
    train_idx = idx[n_val + n_test:]
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def to_xy(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[LABEL_COL].to_numpy(dtype=np.float32)
    return X, y
