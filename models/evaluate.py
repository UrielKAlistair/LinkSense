"""Metrics for AP selection, and the heuristics every model is measured against.

Regression error alone answers the wrong question. What matters operationally
is whether the model, shown a set of APs, picks a good one - so the headline
numbers are decision metrics computed within each scan:

  top1_accuracy  fraction of scans where a highest-scoring option is
                 genuinely among the best. Options whose scores tie share the
                 credit, so a model cannot gain from the order its rows
                 happen to arrive in.
  regret         throughput given up by following the model rather than an
                 oracle that always picks the best option, in Mbps. This is
                 the number to optimise: it is insensitive to harmless ties
                 and denominated in the thing the user cares about.
  regret_frac    the same, divided by the oracle's throughput, so weak and
                 strong scenarios contribute comparably.
  spearman       rank correlation between scores and labels within a scan,
                 averaged - does the model order the whole set sensibly,
                 not just the top of it.

Each metric is also reported averaged per topology, because one topology
contributes several scans and the plain mean would let the topologies
that were simulated more often speak louder.

The same file supplies the reference rules in baseline_predictions(), the most
important of which is strongest_rssi: it is what a commodity client actually
does, and a learned model that cannot beat it has not earned its complexity.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .data import SCAN_COL, LABEL_COL, MISSING_RSSI_SENTINEL

# Throughput difference below which two options count as equally good. Used
# for top-1 credit here and for discarding uninformative pairs in
# ranker.ranking_loss, which imports it so the two cannot drift apart.
TIE_TOL_MBPS = 0.5


def _rank_correlation(pred: np.ndarray, label: np.ndarray) -> float | None:
    """Spearman correlation between scores and labels for one scan.

    Returns None when the scan has no ordering to get right - fewer than
    two options, or every option carrying the same label - so the caller can
    leave it out of the average entirely.

    Returns 0.0 when the labels do differ but every score is identical. That
    case is the model declining to order the set, which earns no credit;
    dropping it instead would hide the ties and report the model only on the
    sets where it did commit to an answer.
    """
    if len(pred) < 2:
        return None
    label_ranks = pd.Series(label).rank().to_numpy()
    if label_ranks.std() == 0:
        return None
    pred_ranks = pd.Series(pred).rank().to_numpy()
    if pred_ranks.std() == 0:
        return 0.0
    return float(np.corrcoef(pred_ranks, label_ranks)[0, 1])


def _per_topology_metrics(topology_ids: list, top1: list, regret: list) -> dict:
    """Re-average per-scan numbers so every topology counts once.

    Empty when the frame carried no topology_id, which is the signal callers
    use to fall back to the plain per-scan mean.
    """
    if not topology_ids or topology_ids[0] is None:
        return {}
    per_set = pd.DataFrame({
        "topology_id": topology_ids,
        "top1": np.asarray(top1, dtype=float),
        "regret": np.asarray(regret, dtype=float),
    })
    per_topology = per_set.groupby("topology_id").mean()
    return {
        "topologies": len(per_topology),
        "topology_top1_accuracy": float(per_topology["top1"].mean()),
        "topology_mean_regret_mbps": float(per_topology["regret"].mean()),
    }


def selection_metrics(df: pd.DataFrame, pred: np.ndarray,
                      tie_tol: float = TIE_TOL_MBPS) -> dict:
    """Decision quality of `pred` used as a per-option score.

    `pred` must be one score per row of `df`, in row order. Options whose
    scores are equal are treated as a uniform random pick among them, so the
    result does not depend on how the rows are sorted.
    """
    pred = np.asarray(pred)
    if len(pred) != len(df):
        raise ValueError(f"prediction length {len(pred)} does not match {len(df)} rows")
    if not np.isfinite(pred).all():
        raise ValueError("predictions contain NaN or infinite values")
    has_topology = "topology_id" in df
    columns = [SCAN_COL, LABEL_COL] + (["topology_id"] if has_topology else [])
    work = df[columns].copy()
    work["pred"] = pred

    top1, regret, regret_frac, rho, topology = [], [], [], [], []
    for _, scan in work.groupby(SCAN_COL):
        y = scan[LABEL_COL].to_numpy()
        p = scan["pred"].to_numpy()
        best = y.max()
        # Every option the score cannot separate from the winner is a
        # possible pick, so score the expectation over picking one of them.
        tied = np.isclose(p, p.max(), rtol=1e-12, atol=1e-12)
        tied_outcomes = y[tied]
        chosen = float(tied_outcomes.mean())
        top1.append(float(np.mean(tied_outcomes >= best - tie_tol)))
        regret.append(best - chosen)
        regret_frac.append((best - chosen) / best if best > 0 else 0.0)
        rho.append(_rank_correlation(p, y))
        topology.append(scan["topology_id"].iloc[0] if has_topology else None)

    scored_rho = [r for r in rho if r is not None]
    metrics = {
        "scans": len(top1),
        "top1_accuracy": float(np.mean(top1)),
        "mean_regret_mbps": float(np.mean(regret)),
        "median_regret_mbps": float(np.median(regret)),
        "mean_regret_frac": float(np.mean(regret_frac)),
        "mean_spearman": float(np.mean(scored_rho)) if scored_rho else float("nan"),
    }
    metrics.update(_per_topology_metrics(topology, top1, regret))
    return metrics


def random_selection_metrics(df: pd.DataFrame, tie_tol: float = TIE_TOL_MBPS) -> dict:
    """The same metrics for a client that picks uniformly at random.

    Computed as an exact expectation over the options rather than by drawing
    one option per scan, because a single draw on a few hundred test
    sets is noisy enough to make random choice look materially good or bad by
    luck alone. Every expectation needed is available from the labels.
    """
    top1, expected_regret, expected_regret_frac, topology = [], [], [], []
    has_topology = "topology_id" in df
    for _, scan in df.groupby(SCAN_COL):
        y = scan[LABEL_COL].to_numpy()
        best = y.max()
        regret = best - y
        top1.append(float(np.mean(y >= best - tie_tol)))
        expected_regret.append(float(regret.mean()))
        expected_regret_frac.append(float(regret.mean() / best) if best > 0 else 0.0)
        topology.append(scan["topology_id"].iloc[0] if has_topology else None)
    metrics = {
        "scans": len(top1),
        "top1_accuracy": float(np.mean(top1)),
        "mean_regret_mbps": float(np.mean(expected_regret)),
        "median_regret_mbps": float(np.median(expected_regret)),
        "mean_regret_frac": float(np.mean(expected_regret_frac)),
        # A random ordering is uncorrelated with the labels in expectation.
        "mean_spearman": 0.0,
    }
    metrics.update(_per_topology_metrics(topology, top1, expected_regret))
    return metrics


def validation_selection_key(frame: pd.DataFrame, pred: np.ndarray) -> tuple[float, float]:
    """Score one candidate configuration on validation; smaller sorts better.

    Every training script in scripts/train/ picks its hyperparameters by
    comparing these tuples, so they all select on the same rule and a change
    here reaches all of them at once.

    The first term is mean regret averaged per topology, so a topology that
    contributed more scans does not get more say; it falls back to the
    plain per-scan mean when the frame carries no topology_id. The second
    term breaks ties on within-scan ordering: regret bottoms out at zero as
    soon as a configuration gets every validation scan right, and on a few
    hundred scans several will, leaving the winner decided by the order the
    grid happens to be written in.
    """
    metrics = selection_metrics(frame, pred)
    regret = metrics.get("topology_mean_regret_mbps", metrics["mean_regret_mbps"])
    spearman = metrics["mean_spearman"]
    # A frame in which no scan has an orderable label leaves spearman
    # undefined; fall back to regret alone rather than sorting on NaN.
    return regret, -spearman if np.isfinite(spearman) else 0.0


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    """Error of `pred` read as a throughput estimate, on linear and log scales.

    r2 is the one that matters if the prediction is to be read as Mbps. r2_log
    is the fairer number for a model trained on a log target or on a ranking
    loss, whose scores may order options perfectly and still score a negative
    linear r2 because squared error in Mbps is dominated by the largest values.
    """
    y = np.asarray(y)
    pred = np.asarray(pred)
    if y.shape != pred.shape:
        raise ValueError(f"label shape {y.shape} does not match prediction shape {pred.shape}")
    if not np.isfinite(pred).all():
        raise ValueError("predictions contain NaN or infinite values")
    resid = y - pred
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ly, lp = np.log1p(np.clip(y, 0, None)), np.log1p(np.clip(pred, 0, None))
    lresid = ly - lp
    lss_tot = float(np.sum((ly - ly.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "r2": float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
        "r2_log": float(1.0 - np.sum(lresid ** 2) / lss_tot) if lss_tot > 0 else float("nan"),
    }


# Candidate values of k for the rssi_minus_busy rule, in dB of signal traded
# against a fully busy channel. The range runs from 0, where the rule collapses
# to strongest_rssi, up to a point well past where adding more occupancy
# penalty starts hurting. A fitted k landing on either end means the optimum
# has moved outside these bounds and the range needs widening.
RSSI_BUSY_K_GRID = tuple(float(k) for k in range(0, 121, 5))


def fit_rssi_busy_k(train_df: pd.DataFrame, busy_column: str,
                    grid: tuple[float, ...] = RSSI_BUSY_K_GRID) -> float:
    """Choose k for the `RSSI - k*busy` rule on TRAINING rows only.

    The heuristic is the bar every learned model has to clear, so it gets its
    one free parameter chosen the same way a model's hyperparameters are:
    by regret, on data the model being compared against never sees scored.
    Raises rather than falling back to a fixed k, because a silently unfitted
    heuristic in one experiment and a fitted one in another would make the
    two experiments' baseline rows incomparable.
    """
    if len(train_df) == 0:
        raise ValueError("cannot fit the rssi_minus_busy rule on an empty frame")
    for column in ("feat_ap_rssi_mean", busy_column, LABEL_COL, SCAN_COL):
        if column not in train_df:
            raise ValueError(f"fit frame is missing {column!r}, needed to fit "
                             "the rssi_minus_busy rule")
    r = train_df["feat_ap_rssi_mean"].fillna(MISSING_RSSI_SENTINEL).to_numpy()
    b = train_df[busy_column].to_numpy()
    scores = [(selection_metrics(train_df, r - k * b)["mean_regret_mbps"], k)
              for k in grid]
    return min(scores)[1]


def busy_column_name(df: pd.DataFrame) -> str | None:
    """Which occupancy column the busy heuristics should read, or None.

    CCA busy is preferred: it counts energy that prevented reception even when
    no packet header was decoded, which is what a real chipset's channel-busy
    counter reports. The decoded-frame airtime column is the fallback for a
    frame-derived corpus that carries no CCA measurement.
    """
    for column in ("feat_chan_cca_busy_frac", "feat_chan_busy_frac"):
        if column in df:
            return column
    return None


def baseline_predictions(df: pd.DataFrame,
                         fit_frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-option scores from the reference rules, for the rows of `df`.

    `fit_frame` supplies the training rows that rssi_minus_busy fits its one
    parameter on; it must not overlap `df`, or that rule is being scored on
    data it was tuned on. Every rule whose input columns are absent from `df`
    is left out of the returned dict rather than faked.
    """
    preds = {}
    # One fixed pseudo-random score per option, derived from its identity so
    # that reordering the rows cannot change it. Only the exported per-row
    # prediction table uses this; headline numbers for random choice come
    # from random_selection_metrics(), which is exact rather than one draw.
    preds["random"] = np.array([
        int.from_bytes(hashlib.blake2b(
            f"{scan}:{option}".encode(), digest_size=8).digest(), "big")
        for scan, option in zip(df[SCAN_COL], df.get("ap_index", df.index))
    ], dtype=np.uint64)
    busy_column = busy_column_name(df)
    if "feat_ap_rssi_mean" in df:
        preds["strongest_rssi"] = (
            df["feat_ap_rssi_mean"].fillna(MISSING_RSSI_SENTINEL).to_numpy())
    if busy_column:
        preds["least_busy_channel"] = -df[busy_column].to_numpy()
    if "feat_ap_rssi_mean" in df and busy_column:
        k = fit_rssi_busy_k(fit_frame, busy_column)
        preds["rssi_minus_busy"] = (preds["strongest_rssi"]
                                    - k * df[busy_column].to_numpy())
    return preds


def evaluate_all(df: pd.DataFrame, model_preds: dict[str, np.ndarray],
                 fit_frame: pd.DataFrame) -> pd.DataFrame:
    """One row per model and per reference rule, scored on the rows of `df`.

    `fit_frame` is handed to baseline_predictions(), so it carries the same
    requirement: training rows, disjoint from `df`.
    """
    y = df[LABEL_COL].to_numpy()
    rows = [{"model": "random", **random_selection_metrics(df)}]
    for name, pred in {**baseline_predictions(df, fit_frame), **model_preds}.items():
        if name == "random":
            continue
        row = {"model": name}
        # The reference rules emit scores on arbitrary scales, not throughput
        # estimates, so an r2 for them would be meaningless.
        if name in model_preds:
            row.update(regression_metrics(y, pred))
        row.update(selection_metrics(df, pred))
        rows.append(row)
    # Any metric not named here still reaches the output, appended after the
    # named ones. A maintainer adding a metric gets it in the table without
    # having to remember to list it.
    preferred = ["model", "top1_accuracy", "mean_regret_mbps", "median_regret_mbps",
                 "mean_regret_frac", "mean_spearman", "topology_top1_accuracy",
                 "topology_mean_regret_mbps", "mae", "rmse", "r2", "r2_log",
                 "scans", "topologies"]
    out = pd.DataFrame(rows)
    ordered = [c for c in preferred if c in out.columns]
    return out[ordered + [c for c in out.columns if c not in ordered]]


def label_spread(df: pd.DataFrame) -> dict:
    """How much throughput the choice itself is worth, before any model.

    Summarises the labels only: what an oracle gets, what the worst option
    gets, and what picking at random gets on average. If the spread between
    best and worst is small the choice barely matters and no model can show a
    useful gain, so this is the first thing to look at for a new corpus.
    """
    g = df.groupby(SCAN_COL)[LABEL_COL]
    best, worst, mean = g.max(), g.min(), g.mean()
    return {
        "mean_best_mbps": float(best.mean()),
        "mean_worst_mbps": float(worst.mean()),
        "mean_random_choice_mbps": float(mean.mean()),
        "mean_spread_mbps": float((best - worst).mean()),
        "scans_all_zero": int((best <= 1e-9).sum()),
    }
