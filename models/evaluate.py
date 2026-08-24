"""Metrics for AP selection.

Regression error alone answers the wrong question. What matters
operationally is whether the model, shown a set of APs, picks a good one -
so the headline metrics are decision metrics computed within each group:

      top1_accuracy  fraction of groups where a maximum-score option is truly
                     best. Exact score ties use their expected uniform
                     tie-break outcome, avoiding accidental AP-index bias.
  regret         throughput given up by following the model instead of the
                 oracle, in Mbps. This is the metric to optimise: it is
                 insensitive to harmless ties and directly denominated in
                 the thing the user cares about.
  regret_frac    the same, normalised by the oracle's throughput, so weak
                 and strong scenarios contribute comparably.
  spearman       rank correlation within groups, averaged - does the model
                 order the whole option set sensibly, not just the top.

Every model is compared against the same baselines, including the
strongest-RSSI rule that a real client actually uses. Beating that rule is
the bar for this project being worth anything.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .data import GROUP_COL, LABEL_COL


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return np.nan
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def selection_metrics(df: pd.DataFrame, pred: np.ndarray,
                      tie_tol: float = 0.5) -> dict:
    """Decision quality of `pred` used as a per-option score.

    tie_tol (Mbps) treats options within that distance of the best as
    equally correct for top-1: picking a 19.9 Mbps AP over a 20.0 Mbps one
    is not a mistake worth counting.
    """
    pred = np.asarray(pred)
    if len(pred) != len(df):
        raise ValueError(f"prediction length {len(pred)} does not match {len(df)} rows")
    if not np.isfinite(pred).all():
        raise ValueError("predictions contain NaN or infinite values")
    columns = [GROUP_COL, LABEL_COL]
    if "topology_id" in df:
        columns.append("topology_id")
    work = df[columns].copy()
    work["pred"] = pred

    top1 = []
    regret = []
    regret_frac = []
    rho = []
    topology = []
    for _, g in work.groupby(GROUP_COL):
        y = g[LABEL_COL].to_numpy()
        p = g["pred"].to_numpy()
        best = y.max()
        tied = np.isclose(p, p.max(), rtol=1e-12, atol=1e-12)
        tied_outcomes = y[tied]
        chosen = float(tied_outcomes.mean())
        top1.append(float(np.mean(tied_outcomes >= best - tie_tol)))
        regret.append(best - chosen)
        regret_frac.append((best - chosen) / best if best > 0 else 0.0)
        rho.append(_spearman(p, y))
        topology.append(g["topology_id"].iloc[0] if "topology_id" in g else None)

    finite_rho = np.asarray(rho)[np.isfinite(rho)]
    metrics = {
        "groups": len(top1),
        "top1_accuracy": float(np.mean(top1)),
        "mean_regret_mbps": float(np.mean(regret)),
        "median_regret_mbps": float(np.median(regret)),
        "mean_regret_frac": float(np.mean(regret_frac)),
        "mean_spearman": float(finite_rho.mean()) if len(finite_rho) else 0.0,
    }
    if "topology_id" in work:
        per_group = pd.DataFrame({
            "topology_id": topology,
            "top1": np.asarray(top1, dtype=float),
            "regret": regret,
        })
        per_topology = per_group.groupby("topology_id").mean()
        metrics.update({
            "topologies": len(per_topology),
            "topology_top1_accuracy": float(per_topology["top1"].mean()),
            "topology_mean_regret_mbps": float(per_topology["regret"].mean()),
        })
    return metrics


def random_selection_metrics(df: pd.DataFrame, tie_tol: float = 0.5) -> dict:
    """Exact expectation for choosing uniformly among each group's options.

    A single pseudo-random draw is unnecessarily noisy on a small test set
    and can make random choice look materially better or worse by luck. Every
    expectation needed here is available analytically from the labels.
    """
    top1 = []
    expected_regret = []
    expected_regret_frac = []
    topology = []
    for _, group in df.groupby(GROUP_COL):
        y = group[LABEL_COL].to_numpy()
        best = y.max()
        regret = best - y
        top1.append(float(np.mean(y >= best - tie_tol)))
        expected_regret.append(float(regret.mean()))
        expected_regret_frac.append(float(regret.mean() / best) if best > 0 else 0.0)
        topology.append(group["topology_id"].iloc[0] if "topology_id" in group else None)
    metrics = {
        "groups": len(top1),
        "top1_accuracy": float(np.mean(top1)),
        "mean_regret_mbps": float(np.mean(expected_regret)),
        "median_regret_mbps": float(np.median(expected_regret)),
        "mean_regret_frac": float(np.mean(expected_regret_frac)),
        # For an independent random ordering, expected rank correlation is 0.
        "mean_spearman": 0.0,
    }
    if "topology_id" in df:
        per_group = pd.DataFrame({
            "topology_id": topology,
            "top1": top1,
            "regret": expected_regret,
        })
        per_topology = per_group.groupby("topology_id").mean()
        metrics.update({
            "topologies": len(per_topology),
            "topology_top1_accuracy": float(per_topology["top1"].mean()),
            "topology_mean_regret_mbps": float(per_topology["regret"].mean()),
        })
    return metrics


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    """Calibration on both the linear and the log scale.

    Reporting only linear r2 would be misleading here: models trained on a
    log target (or predominantly on a scale-free ranking loss) can order
    options almost perfectly and still score a negative linear r2, because
    squared error in Mbps is dominated by the largest values. r2_log is the
    fairer calibration measure for those; r2 is the one that matters if the
    prediction is to be read as a throughput estimate in Mbps.
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


def baseline_predictions(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Reference rules, in increasing order of sophistication.

    strongest_rssi is the one that matters: it is what commodity clients do,
    and a learned model that cannot beat it has not earned its complexity.
    """
    preds = {}
    # This concrete random draw is used only in exported per-row predictions.
    # Stable hashing makes it independent of dataframe row order. Headline
    # evaluation uses random_selection_metrics() instead of this one draw.
    preds["random"] = np.array([
        int.from_bytes(hashlib.blake2b(
            f"{group}:{option}".encode(), digest_size=8).digest(), "big")
        for group, option in zip(df[GROUP_COL], df.get("ap_index", df.index))
    ], dtype=np.uint64)
    if "feat_ap_rssi_mean" in df:
        preds["strongest_rssi"] = df["feat_ap_rssi_mean"].fillna(-120).to_numpy()
    # CCA busy includes energy that prevented reception even when no packet
    # header was decoded. That is both closer to a real chipset's channel-busy
    # counter and the reason Simulator 1 records chanbusy.csv. The decoded-frame
    # airtime feature remains a fallback so old datasets can still be inspected.
    busy_column = ("feat_chan_cca_busy_frac"
                   if "feat_chan_cca_busy_frac" in df else
                   "feat_chan_busy_frac" if "feat_chan_busy_frac" in df else None)
    if busy_column:
        preds["least_busy_channel"] = -df[busy_column].to_numpy()
    if "feat_ap_rssi_mean" in df and busy_column:
        # crude hand-tuned mix of the two signals, as a "sensible heuristic"
        r = df["feat_ap_rssi_mean"].fillna(-120).to_numpy()
        b = df[busy_column].to_numpy()
        preds["rssi_minus_busy"] = (r + 100.0) / 40.0 - b
    return preds


def evaluate_all(df: pd.DataFrame, model_preds: dict[str, np.ndarray]) -> pd.DataFrame:
    y = df[LABEL_COL].to_numpy()
    rows = [{"model": "random", **random_selection_metrics(df)}]
    for name, pred in {**baseline_predictions(df), **model_preds}.items():
        if name == "random":
            continue
        row = {"model": name}
        # baselines are scores, not throughput estimates, so regression
        # metrics are only meaningful for the actual regressors
        if name in model_preds:
            row.update(regression_metrics(y, pred))
        row.update(selection_metrics(df, pred))
        rows.append(row)
    # Preferred ordering first, then anything else that turned up. Listing
    # the columns exhaustively meant a newly added metric was silently
    # dropped here and only surfaced as a KeyError much later.
    preferred = ["model", "top1_accuracy", "mean_regret_mbps", "median_regret_mbps",
                 "mean_regret_frac", "mean_spearman", "topology_top1_accuracy",
                 "topology_mean_regret_mbps", "mae", "rmse", "r2", "r2_log",
                 "groups", "topologies"]
    out = pd.DataFrame(rows)
    ordered = [c for c in preferred if c in out.columns]
    return out[ordered + [c for c in out.columns if c not in ordered]]


def oracle_ceiling(df: pd.DataFrame) -> dict:
    """How much is on the table: the spread a perfect chooser would capture
    over a random one. If this is small the task is not worth modelling."""
    g = df.groupby(GROUP_COL)[LABEL_COL]
    best, worst, mean = g.max(), g.min(), g.mean()
    return {
        "mean_best_mbps": float(best.mean()),
        "mean_worst_mbps": float(worst.mean()),
        "mean_random_choice_mbps": float(mean.mean()),
        "mean_spread_mbps": float((best - worst).mean()),
        "groups_all_zero": int((best <= 1e-9).sum()),
    }
