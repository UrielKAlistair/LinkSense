"""Dataset loading, splitting and leakage control for AP selection.

The dataset is a choice set: rows sharing a group_id describe one physical
situation, one row per AP the client could join. Two rules follow from that
and are enforced here rather than left to each model:

  1. Only feat_* columns may be model inputs. gt_* columns are simulator
     ground truth (true distance, true offered load, seeds) that a real
     station cannot observe before associating; label_* is the answer.
     feature_columns() is the single definition of "what the model sees".

  2. Splits are by GROUP, never by row. Rows in a group share one
     pre-association observation, so splitting rows would put near-copies
     of a test observation into training and report an optimistic score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_COL = "label_throughput_mbps"
GROUP_COL = "group_id"
OPTION_COL = "ap_index"

# "Heard nothing from this AP" is not a small RSSI, it is a different kind of
# observation. Models get an explicit flag (feat_rel_seen) plus this
# out-of-range sentinel, so they can separate "weak" from "absent" instead of
# treating an absent AP as one sitting at the bottom of the scale.
MISSING_RSSI_SENTINEL = -100.0


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("feat_")]


def ground_truth_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("gt_")]


def load_dataset(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = {LABEL_COL, GROUP_COL, OPTION_COL} - set(df.columns)
    if missing:
        raise ValueError(f"dataset missing required columns: {missing}")
    return df


def impute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Give absent-AP NaNs a defined value so any model can consume them.

    Tree ensembles could take the NaNs directly, but the MLP cannot, and
    using one imputed frame everywhere keeps the comparison honest.
    """
    df = df.copy()
    feats = feature_columns(df)
    rssi_like = [c for c in feats if "rssi" in c]
    for c in rssi_like:
        df[c] = df[c].fillna(MISSING_RSSI_SENTINEL)
    for c in (c for c in feats if c not in rssi_like):
        df[c] = df[c].fillna(0.0)
    return df


def split_by_group(df: pd.DataFrame, val_frac: float = 0.2, test_frac: float = 0.2,
                   seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition whole groups, so no observation spans two splits."""
    groups = df[GROUP_COL].unique()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(groups))
    n_val = int(len(groups) * val_frac)
    n_test = int(len(groups) * test_frac)
    sets = {
        "val": set(groups[perm[:n_val]]),
        "test": set(groups[perm[n_val:n_val + n_test]]),
        "train": set(groups[perm[n_val + n_test:]]),
    }
    out = tuple(df[df[GROUP_COL].isin(sets[k])].reset_index(drop=True)
                for k in ("train", "val", "test"))
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"])
    return out


def to_xy(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[LABEL_COL].to_numpy(dtype=np.float32)
    return X, y


def assert_no_leakage(feature_cols: list[str]) -> None:
    bad = [c for c in feature_cols if not c.startswith("feat_")]
    if bad:
        raise AssertionError(f"non-observable columns used as features: {bad}")
