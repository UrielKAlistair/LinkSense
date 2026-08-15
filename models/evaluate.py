"""Metrics for AP selection.

Regression error alone answers the wrong question. What matters
operationally is whether the model, shown a set of APs, picks a good one -
so the headline metrics are decision metrics computed within each group:

  top1_accuracy  fraction of groups where the argmax of the prediction is
                 the truly best AP. Blunt: near-ties count as failures even
                 when the cost of picking either is nil.
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
    work = df[[GROUP_COL, LABEL_COL]].copy()
    work["pred"] = pred

    top1 = []
    regret = []
    regret_frac = []
    rho = []
    for _, g in work.groupby(GROUP_COL):
        y = g[LABEL_COL].to_numpy()
        p = g["pred"].to_numpy()
        best = y.max()
        chosen = y[int(np.argmax(p))]
        top1.append(bool(chosen >= best - tie_tol))
        regret.append(best - chosen)
        regret_frac.append((best - chosen) / best if best > 0 else 0.0)
        rho.append(_spearman(p, y))

    return {
        "groups": len(top1),
        "top1_accuracy": float(np.mean(top1)),
        "mean_regret_mbps": float(np.mean(regret)),
        "median_regret_mbps": float(np.median(regret)),
        "mean_regret_frac": float(np.mean(regret_frac)),
        "mean_spearman": float(np.nanmean(rho)),
    }


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    resid = y - pred
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "r2": float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
    }


def baseline_predictions(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Reference rules, in increasing order of sophistication.

    strongest_rssi is the one that matters: it is what commodity clients do,
    and a learned model that cannot beat it has not earned its complexity.
    """
    preds = {}
    preds["random"] = np.random.default_rng(0).normal(size=len(df))
    if "feat_ap_rssi_mean" in df:
        preds["strongest_rssi"] = df["feat_ap_rssi_mean"].fillna(-120).to_numpy()
    if "feat_chan_busy_frac" in df:
        preds["least_busy_channel"] = -df["feat_chan_busy_frac"].to_numpy()
    if {"feat_ap_rssi_mean", "feat_chan_busy_frac"} <= set(df.columns):
        # crude hand-tuned mix of the two signals, as a "sensible heuristic"
        r = df["feat_ap_rssi_mean"].fillna(-120).to_numpy()
        b = df["feat_chan_busy_frac"].to_numpy()
        preds["rssi_minus_busy"] = (r + 100.0) / 40.0 - b
    return preds


def evaluate_all(df: pd.DataFrame, model_preds: dict[str, np.ndarray]) -> pd.DataFrame:
    y = df[LABEL_COL].to_numpy()
    rows = []
    for name, pred in {**baseline_predictions(df), **model_preds}.items():
        row = {"model": name}
        # baselines are scores, not throughput estimates, so regression
        # metrics are only meaningful for the actual regressors
        if name in model_preds:
            row.update(regression_metrics(y, pred))
        row.update(selection_metrics(df, pred))
        rows.append(row)
    cols = ["model", "top1_accuracy", "mean_regret_mbps", "median_regret_mbps",
            "mean_regret_frac", "mean_spearman", "mae", "rmse", "r2", "groups"]
    out = pd.DataFrame(rows)
    return out[[c for c in cols if c in out.columns]]


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
