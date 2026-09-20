"""Split the scans into training, validation and test parts.

A topology is one simulated deployment: its APs, their channels and their
traffic. The client is placed at several spots in it, and each spot is one
scan, so the scans of a topology hear the same APs and the same traffic and are
near-copies of one another. Every scan of a topology lands in the same part.

We also want each part to hold the corpus's mix of deployment sizes, so the
topologies are dealt one AP count at a time and no part draws more than its
share of the large deployments.

PROCESS, for one value of test_fold
  1. Reduce the rows to one entry per topology, each paired with its AP count.
     split_by_topology() takes the aggregate table, one row per valid AP;
     split_rows_by_topology() takes the cell cache's per-scan arrays.
  2. Deal those topologies into n_folds parts under one fixed permutation, each
     AP count dealt separately.
  3. Hold part test_fold out for test and the next one along for validation,
     and train on the rest. At five folds a run sees 60/20/20, and over the
     five runs every topology is tested exactly once and never while it is
     being trained on.

Both entry points hand their topologies to the same rotation, so for a given
test_fold they put the same topologies in the same parts, and both return
train, validation and test in that order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# TOPOLOGY_COL names the unit every split here keeps whole, and N_APS_COL the
# column we stratify on.
TOPOLOGY_COL = "topology_id"
N_APS_COL = "gt_n_aps"

# How many parts the corpus is dealt into, and so how many runs it takes to
# test every topology once. Below MIN_FOLDS a rotation leaves no training part.
N_FOLDS = 5
MIN_FOLDS = 3

# Seeds the permutation that deals topologies into folds.
DEAL_SEED = 0


# ---------------------------------------------------------------------------
# 1. From rows to topologies
# ---------------------------------------------------------------------------

# One entry point for the aggregate table, one for the npz cache.

def split_by_topology(df: pd.DataFrame, test_fold: int = 0, n_folds: int = N_FOLDS
                      ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The three parts of the aggregate table, one row per valid AP.

    Every row of a topology lands in exactly one part.
    """
    if TOPOLOGY_COL not in df.columns:
        raise ValueError(
            f"dataset has no {TOPOLOGY_COL} column, so it cannot be split without "
            "risking repeated seeds of one deployment landing in two parts")
    if N_APS_COL not in df.columns:
        raise ValueError(f"dataset has no {N_APS_COL} column, so the folds cannot "
                         "be given the same mix of deployment sizes")
    missing = int(df[TOPOLOGY_COL].isna().sum())
    if missing:
        raise ValueError(f"{missing} rows have no {TOPOLOGY_COL}; they would be "
                         "dropped from every part rather than assigned to one")

    # partition_ids splits topologies, and df holds one row per valid AP, so a
    # topology appears once per AP per scan. Sorted because the deal is
    # positional: the parts must not follow the order the rows arrived in.
    topologies = (df[[TOPOLOGY_COL, N_APS_COL]].drop_duplicates(TOPOLOGY_COL)
                  .sort_values(TOPOLOGY_COL).reset_index(drop=True))

    parts = partition_ids(topologies[TOPOLOGY_COL].to_numpy(),
                          topologies[N_APS_COL].to_numpy(), test_fold, n_folds)
    return tuple(df[df[TOPOLOGY_COL].isin(set(part))].reset_index(drop=True)
                 for part in parts)


def split_rows_by_topology(topology_ids: np.ndarray, n_aps: np.ndarray,
                           test_fold: int = 0, n_folds: int = N_FOLDS
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three parts as positions into a per-scan axis in the npz cache.

    `topology_ids` and `n_aps` hold one entry per scan: the topology that scan
    was taken in, and how many APs that deployment has.
    """
    topology_ids = np.asarray(topology_ids)

    # The reduction drop_duplicates does above, in numpy: one entry per
    # topology, paired with its AP count. Every scan of a topology reports the
    # same count, so the first scan of each is enough to read it from.
    topologies, first_scan = np.unique(topology_ids, return_index=True)
    topology_n_aps = np.asarray(n_aps)[first_scan]

    parts = partition_ids(topologies, topology_n_aps, test_fold, n_folds)

    return tuple(np.flatnonzero(np.isin(topology_ids, part)) for part in parts)


# ---------------------------------------------------------------------------
# 2. The deal, and one rotation of it
# ---------------------------------------------------------------------------

# Where fold membership is settled. The deal reads the identifiers and their AP
# counts and nothing else, so which fold a topology falls in is a property of
# the corpus rather than of the model being trained on it. It is the same deal
# on every call: test_fold only chooses which of its parts to hand back as what.

def partition_ids(ids: np.ndarray, n_aps: np.ndarray, test_fold: int,
                  n_folds: int = N_FOLDS
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train, validation and test identifiers for one rotation of the folds.

    `ids` are the units being split - topology identifiers everywhere in this
    repo - and `n_aps` the AP count of each, balanced across the folds.
    """
    ids = np.asarray(ids)
    n_aps = np.asarray(n_aps)
    if n_folds < MIN_FOLDS:
        raise ValueError(f"at least {MIN_FOLDS} folds are needed to leave a "
                         f"training part; asked for {n_folds}")
    if not 0 <= test_fold < n_folds:
        raise ValueError(f"fold {test_fold} is not one of the {n_folds} folds")
    if pd.isna(n_aps).any():
        raise ValueError("an AP count is NaN; every identifier needs one to be split")
    ap_counts, per_count = np.unique(n_aps, return_counts=True)
    rarest = int(per_count.min()) if per_count.size else 0
    if rarest < n_folds:
        raise ValueError(
            f"every AP count needs at least {n_folds} topologies to reach every "
            f"fold; the rarest has {rarest}")

    # Deal into n_folds parts of near-equal size: the identifiers of one AP
    # count are permuted under DEAL_SEED and cut into n_folds slices, and every
    # fold takes one slice of every count. array_split hands the remainder of a
    # count that does not divide evenly to the earliest folds.
    rng = np.random.default_rng(DEAL_SEED)
    folds = [[] for _ in range(n_folds)]
    for count in ap_counts:
        members = ids[n_aps == count]
        for bucket, slice_ in zip(folds, np.array_split(
                members[rng.permutation(len(members))], n_folds)):
            bucket.extend(slice_)
    folds = [np.asarray(bucket) for bucket in folds]

    val_fold = (test_fold + 1) % n_folds
    train = np.concatenate([bucket for other, bucket in enumerate(folds)
                            if other not in (test_fold, val_fold)])
    return train, folds[val_fold], folds[test_fold]
