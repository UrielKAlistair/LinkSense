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

# The dataset builder excludes APs that were not discovered during the scan,
# but RSSI-derived values can still be missing when a statistic needs more
# samples than the scan captured. This out-of-range level remains a defensive
# imputation value for absolute dBm columns.
MISSING_RSSI_SENTINEL = -100.0


def _is_rssi_level(col: str) -> bool:
    """True only for columns that are an absolute signal level in dBm.

    The sentinel above is well formed for a LEVEL: the simulated noise floor
    is about -94 dBm, so -100 is off the bottom of the physical scale and
    points in the right direction.

    None of that holds for the other columns whose names happen to contain
    "rssi". A standard deviation lives in [0, inf), so -100 is not extreme
    there but impossible. A margin or a difference has a real range of
    roughly +/-40 dB, so -100 asserts a measurement that never happened and
    is 2.5x larger than anything real - which then dominates the standardiser
    the MLP fits, squashing every genuine value into a fraction of a sigma.
    Those get a neutral 0.0 instead.
    """
    if "rssi" not in col:
        return False
    return not any(k in col for k in
                   ("_std", "margin", "minus", "rank", "share", "_frac"))


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
    """Give missing feature values a defined value so any model can consume them.

    Tree ensembles could take the NaNs directly, but the MLP cannot, and
    using one imputed frame everywhere keeps the comparison honest.
    """
    df = df.copy()
    feats = feature_columns(df)
    levels = [c for c in feats if _is_rssi_level(c)]
    for c in levels:
        df[c] = df[c].fillna(MISSING_RSSI_SENTINEL)
    for c in (c for c in feats if c not in levels):
        df[c] = df[c].fillna(0.0)
    return df


def split_by_group(df: pd.DataFrame, val_frac: float = 0.2, test_frac: float = 0.2,
                   seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition whole choice sets, and repeated seeds of a topology together."""
    split_col = "topology_id" if "topology_id" in df.columns else GROUP_COL
    group_frame = df.groupby(split_col, sort=True).first().reset_index()
    groups = group_frame[split_col].to_numpy()
    if len(groups) < 5:
        raise ValueError(
            f"at least 5 independent {split_col} values are required for a 60/20/20 split; "
            f"found {len(groups)}")
    strata = group_frame["gt_n_aps"].to_numpy() if "gt_n_aps" in group_frame else None
    train_groups, val_groups, test_groups = partition_groups(
        groups, strata, val_frac, test_frac, seed)
    sets = {"train": set(train_groups), "val": set(val_groups), "test": set(test_groups)}
    out = tuple(df[df[split_col].isin(sets[k])].reset_index(drop=True)
                for k in ("train", "val", "test"))
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"])
    return out


def partition_groups(groups: np.ndarray, strata: np.ndarray | None,
                     val_frac: float, test_frac: float, seed: int
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split group identifiers, stratifying only when every stratum is viable."""
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    if strata is not None:
        strata = np.asarray(strata)
        counts = pd.Series(strata).value_counts()
    else:
        counts = pd.Series(dtype=int)

    # Five members are the minimum that can place one in validation, one in
    # test, and retain a training majority. Tiny pilots use the unstratified
    # fallback rather than deleting a rare AP count from training.
    if len(counts) and counts.min() >= 5:
        partitions = {"train": [], "val": [], "test": []}
        for value in sorted(counts.index):
            members = groups[strata == value]
            members = members[rng.permutation(len(members))]
            n_val = max(1, int(round(len(members) * val_frac)))
            n_test = max(1, int(round(len(members) * test_frac)))
            partitions["val"].extend(members[:n_val])
            partitions["test"].extend(members[n_val:n_val + n_test])
            partitions["train"].extend(members[n_val + n_test:])
        return tuple(np.asarray(partitions[name]) for name in ("train", "val", "test"))

    permuted = groups[rng.permutation(len(groups))]
    n_val = int(len(groups) * val_frac)
    n_test = int(len(groups) * test_frac)
    return (permuted[n_val + n_test:], permuted[:n_val],
            permuted[n_val:n_val + n_test])


def to_xy(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[LABEL_COL].to_numpy(dtype=np.float32)
    return X, y


def group_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """Per-row weights giving every AP-choice decision equal total weight.

    Without this, a group with seven discovered APs contributes 3.5 times the
    regression loss of a two-option group even though both count as one
    decision in the headline metrics. Scaling to mean one keeps estimator
    regularisation parameters on their usual numerical scale.
    """
    group_size = df.groupby(GROUP_COL)[GROUP_COL].transform("size").to_numpy()
    weights = 1.0 / group_size
    return (weights / weights.mean()).astype(np.float64)


def assert_no_leakage(feature_cols: list[str]) -> None:
    bad = [c for c in feature_cols if not c.startswith("feat_")]
    if bad:
        raise AssertionError(f"non-observable columns used as features: {bad}")
