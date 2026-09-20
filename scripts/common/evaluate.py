"""Metrics for AP selection and throughput prediction. All models are evaluated 
on these metrics.

We use two sets of metrics: regression_metrics(), a direct measure of the distance
between the true and predicted throughputs and selection_metrics(), which looks
at the goodness of the final AP choice and the relative ordering of APs.

  regret         throughput given up by following the model's choice instead of an
                 oracle that always picks the best AP, in Mbps. 
  regret_frac    the same, divided by the oracle's throughput.

  top1_rate      how often is the model's choice the best AP, counting a tie as
                 a chance of landing on one.
  regret_cdf_*   Prob(Regret <= epsilon) is the CDF. This is called with some
                 margin value epsilon and outputs the percentage of the predictions
                 that have only that margin of error from the oracle's optimum.

  spearman       rank correlation between predictions and labels within a
                 scan, averaged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Columns of the per-AP frame these metrics score, one row per valid AP.
LABEL_COL = "label_throughput_mbps"
SCAN_COL = "scan_id"

# Margins the regret CDF is read at, as a share of the oracle's throughput.
# Each one becomes a regret_cdf_*pct column.
REGRET_CDF_MARGINS = (0.01, 0.05, 0.10)


def all_metrics(df: pd.DataFrame, pred: np.ndarray) -> dict:
    """Every metric for one model, whose `pred` is a throughput estimate in Mbps,
    one per row of `df`."""
    return {**regression_metrics(df[LABEL_COL].to_numpy(), pred),
            **selection_metrics(df, pred)}


# ---------------------------------------------------------------------------
# Regression: how close each prediction is to its label
# ---------------------------------------------------------------------------

def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    """Error of `pred` read as a throughput estimate, on linear and log scales.
    """
    y = np.asarray(y)
    pred = np.asarray(pred)
    if y.shape != pred.shape:
        raise ValueError(f"label shape {y.shape} does not match prediction shape {pred.shape}")
    if not np.isfinite(pred).all():
        raise ValueError("predictions contain NaN or infinite values")
    if not np.isfinite(y).all():
        raise ValueError("labels contain NaN or infinite values")
        
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


# ---------------------------------------------------------------------------
# Selection: what joining the top-scored AP costs
# ---------------------------------------------------------------------------

def selection_metrics(df: pd.DataFrame, pred: np.ndarray) -> dict:
    """How good a choice `pred` makes, read as a score for each candidate AP.

    Two columns of `df` are read and no others. scan_id groups the rows into the
    candidate sets a client actually chose between, and label_throughput_mbps
    says what each candidate went on to deliver. Every metric here is defined
    inside one scan - regret is measured against the best AP that scan could
    see - so the grouping is what the frame is for, and a bare vector of labels
    could not be grouped.

    `pred` is one score per row, in row order. Only its ordering within a scan
    is used, never its scale, so a rule scoring in dBm and a model predicting
    Mbps are judged the same way.
    """
    pred = np.asarray(pred)
    if len(pred) != len(df):
        raise ValueError(f"prediction length {len(pred)} does not match {len(df)} rows")
    if not np.isfinite(pred).all():
        raise ValueError("predictions contain NaN or infinite values")

    work = df[[SCAN_COL, LABEL_COL]].copy()
    work["pred"] = pred

    regret, regret_frac, hit, rho = [], [], [], []
    for _, scan in work.groupby(SCAN_COL):
        label = scan[LABEL_COL].to_numpy()
        score = scan["pred"].to_numpy()
        best = label.max()

        # The client joins whichever AP the score ranks first. Where the score
        # cannot separate the leaders it joins one of them at random, so each
        # measure below takes the expectation over that tie rather than letting
        # row order pick a winner.
        leaders = np.isclose(score, score.max(), rtol=1e-12, atol=1e-12)
        joined = float(label[leaders].mean())

        # What the choice cost: in Mbps, and against what was on offer.
        regret.append(best - joined)
        regret_frac.append((best - joined) / best if best > 0 else 0.0)

        # Whether the choice was right at all, rather than what it cost.
        hit.append(float(np.isclose(label[leaders], best,
                                    rtol=1e-12, atol=1e-12).mean()))

        # Whether the whole set was ordered well, not just its winner.
        rho.append(_rank_correlation(score, label))

    # A scan with nothing to order leaves its correlation undefined, and drops
    # out of that average alone.
    scored_rho = [r for r in rho if r is not None]

    return {
        "top1_rate": float(np.mean(hit)),
        **{f"regret_cdf_{round(margin * 100)}pct": regret_cdf(regret_frac, margin)
           for margin in REGRET_CDF_MARGINS},
        "mean_regret_mbps": float(np.mean(regret)),
        "median_regret_mbps": float(np.median(regret)),
        "mean_regret_frac": float(np.mean(regret_frac)),
        "mean_spearman": float(np.mean(scored_rho)) if scored_rho else float("nan"),
        "scans": len(regret),
    }


def regret_cdf(regret_frac: list[float], margin: float) -> float:
    """Share of scans that gave up no more than `margin` of the oracle.

    One point of the distribution of regret, rather than a summary of it:
    top1_rate is this at a margin of zero, mean_regret_mbps is its first moment.
    A rate that climbs steeply as the margin widens says the score was usually
    close; one that stays flat says the misses were not close at all.
    """
    return float(np.mean(np.asarray(regret_frac) <= margin + 1e-12))


def _rank_correlation(pred: np.ndarray, label: np.ndarray) -> float | None:
    """Spearman correlation between scores and labels for one scan.

    Returns None when the scan has no ordering to get right - fewer than
    two APs, or every AP carrying the same label - so the caller can
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


def validation_selection_key(frame: pd.DataFrame, pred: np.ndarray) -> tuple[float, float]:
    """Score one candidate configuration on validation; smaller sorts better.

    The first term is mean regret over the validation scans. The second breaks
    ties on within-scan ordering: regret bottoms out at zero as soon as
    a configuration gets every validation scan right, and on a few hundred
    scans several will, leaving the winner decided by the order the grid
    happens to be written in.
    """
    metrics = selection_metrics(frame, pred)
    regret = metrics["mean_regret_mbps"]
    spearman = metrics["mean_spearman"]
    # A frame in which no scan has an orderable label leaves spearman
    # undefined; fall back to regret alone rather than sorting on NaN.
    return regret, -spearman if np.isfinite(spearman) else 0.0


# ---------------------------------------------------------------------------
# Before any model
# ---------------------------------------------------------------------------

def label_spread(df: pd.DataFrame) -> dict:
    """How much throughput the choice itself is worth, before any model.

    Summarises the labels only: what an oracle gets, what the worst AP
    gets, and what picking at random gets on average. If the spread between
    best and worst is small the choice barely matters and no model can show a
    useful gain.
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
