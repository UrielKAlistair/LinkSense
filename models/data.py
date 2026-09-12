"""Dataset loading, splitting and leakage control for AP selection.

Rows sharing a scan_id are one decision: one row per AP the client could have
joined, from one place it stood. Two rules follow, enforced here rather than
left to each model:

  1. Only feat_* columns may be model inputs. gt_* columns are simulator
     ground truth (true distance, true offered load, seeds) a real station
     cannot observe before associating; label_* is the answer.
     feature_columns() is the single definition of "what the model sees".

  2. Splits are by TOPOLOGY, never by row and never by scan. One topology is
     scanned several times over the same deployment, so its scans are
     near-copies; putting one in training and another in test reports a score
     inflated by memorisation rather than prediction. split_by_topology() and
     split_rows_by_topology() are the only entry points that produce a split,
     and neither falls back to a finer unit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_COL = "label_throughput_mbps"
SCAN_COL = "scan_id"
OPTION_COL = "ap_index"
TOPOLOGY_COL = "topology_id"

# Fewest topologies that can put one in validation, one in test and still
# leave a training majority. Below this a split is not meaningful and both
# split entry points refuse rather than return a degenerate partition.
MIN_TOPOLOGIES = 5

# Fill value for an absolute signal level (dBm) that the scan did not
# measure. Filling a level with 0.0 like every other column would assert the
# strongest signal physically possible and make the missing AP the preferred
# choice; this level sits below the simulated noise floor of about -94 dBm,
# so a missing AP sorts last instead. It is the only out-of-range level in
# the repo - evaluate.py fills the same columns with the same value.
MISSING_RSSI_SENTINEL = -100.0


def _is_rssi_level(col: str) -> bool:
    """True for feature columns holding an absolute signal level in dBm.

    Excluded are the other columns whose names contain "rssi" but whose
    values are not levels: a standard deviation (never negative), and the
    margins, differences, ranks and shares that compare one AP against the
    others in its scan. Filling any of those with the level sentinel
    would insert a value far outside their real range, which then dominates
    the standardiser that ranker.py fits.

    Maintainers adding a feat_*rssi* column must check which side it falls
    on: a new name matching none of the excluded keywords is treated as a
    level by default.
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
    missing = {LABEL_COL, SCAN_COL, OPTION_COL} - set(df.columns)
    if missing:
        raise ValueError(f"dataset missing required columns: {missing}")
    return df


def impute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy in which no feat_* column contains NaN.

    Signal levels are filled with MISSING_RSSI_SENTINEL and every other
    feature with 0.0. Tree ensembles accept NaN and the neural models do
    not, so imputing once here means every model is fitted on identical
    inputs. Callers that skip this step and hand a NaN to ranker.py or
    temporal.py get NaN scores and a metric that refuses them.
    """
    df = df.copy()
    feats = feature_columns(df)
    levels = [c for c in feats if _is_rssi_level(c)]
    for c in levels:
        df[c] = df[c].fillna(MISSING_RSSI_SENTINEL)
    for c in (c for c in feats if c not in levels):
        df[c] = df[c].fillna(0.0)
    return df


def split_by_topology(df: pd.DataFrame, val_frac: float = 0.2, test_frac: float = 0.2,
                      seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition a flat dataframe into train/val/test whole topologies.

    Every row of a topology lands in exactly one part, so no repeated seed of
    a deployment is ever split across the boundary. Raises if the frame has
    no topology_id: a frame that cannot be split safely must fail loudly
    rather than be split by some finer unit.
    """
    if TOPOLOGY_COL not in df.columns:
        raise ValueError(
            f"dataset has no {TOPOLOGY_COL} column, so it cannot be split without "
            "risking repeated seeds of one deployment landing in two parts")
    topology_frame = df.groupby(TOPOLOGY_COL, sort=True).first().reset_index()
    topologies = topology_frame[TOPOLOGY_COL].to_numpy()
    if len(topologies) < MIN_TOPOLOGIES:
        raise ValueError(
            f"at least {MIN_TOPOLOGIES} independent {TOPOLOGY_COL} values are "
            f"required for a 60/20/20 split; found {len(topologies)}")
    strata = (topology_frame["gt_n_aps"].to_numpy()
              if "gt_n_aps" in topology_frame else None)
    parts = partition_ids(topologies, strata, val_frac, test_frac, seed)
    _assert_disjoint(parts)
    return tuple(df[df[TOPOLOGY_COL].isin(set(part))].reset_index(drop=True)
                 for part in parts)


def split_rows_by_topology(topology_ids: np.ndarray, strata: np.ndarray | None,
                           val_frac: float = 0.2, test_frac: float = 0.2,
                           seed: int = 0
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The same split for an array corpus: positions into a per-scan axis.

    `topology_ids` and `strata` are both indexed by scan, so a topology that
    appears in several scans contributes each of them to whichever part it
    lands in. Returns train, val and test positions into that axis, which is
    what temporal.py and frames.py index their arrays by.
    """
    topology_ids = np.asarray(topology_ids)
    topologies = np.unique(topology_ids)
    if len(topologies) < MIN_TOPOLOGIES:
        raise ValueError(
            f"at least {MIN_TOPOLOGIES} independent topologies are required for "
            f"a 60/20/20 split; found {len(topologies)}")
    per_topology_strata = None
    if strata is not None:
        strata = np.asarray(strata)
        first_group = [np.flatnonzero(topology_ids == t)[0] for t in topologies]
        per_topology_strata = strata[first_group]
    parts = partition_ids(topologies, per_topology_strata, val_frac, test_frac, seed)
    _assert_disjoint(parts)
    return tuple(np.flatnonzero(np.isin(topology_ids, part)) for part in parts)


def _assert_disjoint(parts: tuple[np.ndarray, ...]) -> None:
    """Fail if any identifier reached two parts.

    partition_ids builds disjoint parts by construction, so this only fires
    if that function is changed incorrectly - which is the one bug in this
    file that would not show up as an error anywhere, only as scores that
    are too good. Kept as a real check rather than an assert so that running
    under python -O cannot switch it off.
    """
    names = ("train", "val", "test")
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            shared = set(parts[i]) & set(parts[j])
            if shared:
                raise AssertionError(
                    f"{names[i]} and {names[j]} share {len(shared)} identifiers: "
                    f"{sorted(shared)[:5]}")


def partition_ids(ids: np.ndarray, strata: np.ndarray | None,
                  val_frac: float, test_frac: float, seed: int
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle identifiers into train/val/test, stratifying when it is viable.

    `ids` are the units being split - topology identifiers everywhere in this
    repo - and `strata` is a parallel array of the value to balance across
    the three parts, or None to skip balancing. Stratification is skipped
    when any stratum has fewer than MIN_TOPOLOGIES members, because taking
    one for validation and one for test out of a stratum that small leaves
    too little of it to train on.
    """
    ids = np.asarray(ids)
    rng = np.random.default_rng(seed)
    counts = (pd.Series(np.asarray(strata)).value_counts()
              if strata is not None else pd.Series(dtype=int))

    def cut(members: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        members = members[rng.permutation(len(members))]
        n_val = max(1, round(len(members) * val_frac))
        n_test = max(1, round(len(members) * test_frac))
        return (members[n_val + n_test:], members[:n_val],
                members[n_val:n_val + n_test])

    if len(counts) and counts.min() >= MIN_TOPOLOGIES:
        strata = np.asarray(strata)
        parts = ([], [], [])
        for value in sorted(counts.index):
            for part, members in zip(parts, cut(ids[strata == value])):
                part.extend(members)
        return tuple(np.asarray(part) for part in parts)
    return cut(ids)


def to_xy(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[LABEL_COL].to_numpy(dtype=np.float32)
    return X, y


def scan_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """Per-row weights giving every AP-choice decision equal total weight.

    A row's weight is the reciprocal of its scan's size, so an
    eight-option scan and a two-option scan contribute the same total to a
    regression loss even though one has four times the rows - matching the
    headline metrics, which count each scan once. The weights are then
    scaled to mean one, which keeps an estimator's regularisation parameters
    on the numerical scale they were tuned for.
    """
    group_size = df.groupby(SCAN_COL)[SCAN_COL].transform("size").to_numpy()
    weights = 1.0 / group_size
    return (weights / weights.mean()).astype(np.float64)


def assert_no_leakage(feature_cols: list[str]) -> None:
    bad = [c for c in feature_cols if not c.startswith("feat_")]
    if bad:
        raise AssertionError(f"non-observable columns used as features: {bad}")
