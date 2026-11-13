#!/usr/bin/env python3
"""Where throughput prediction is hard, and what a tabular model needs to do it.

INPUT   the aggregate table of build_datatable.py: one row per labelled AP,
        with feat_* inputs, the label, and the gt_* regime of its topology.
OUTPUT  <out-dir>/regime_analysis.json and the same numbers printed.

PROCESS, every part on the topology folds of common/splits.py:
  1. Reference errors: the training median, and a rule with nothing fitted.
  2. One gradient-boosted model per feature group, to see which inputs carry
     the prediction.
  3. That model's error inside each third of every regime parameter and each
     band of the label.
  4. Held-out regimes: fit on the lower half of a regime parameter and test on
     the upper half, and the reverse, beside a fit on as many topologies drawn
     from the whole range.
  5. A learning curve over the share of training topologies.
  6. The three factors of the label, each predicted on its own: the share of
     time the candidate transmitted, the PHY rate it used and the share of its
     frames that were acknowledged.
  7. Privileged facts: the same model given what the simulator knows and the
     client cannot observe directly, to bound what the observation leaves out.

Run:
  .venv/bin/python3 scripts/baselines/regime_analysis.py data/aggregate_v2.csv \
      --runs-dir data/runs_v2 --out-dir results/v2/regime_analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.common.evaluate import LABEL_COL  # noqa: E402
from scripts.common.splits import N_FOLDS, TOPOLOGY_COL, split_by_topology  # noqa: E402

REGIMES = ("gt_tcp_fraction", "gt_downlink_fraction", "gt_load_median_mbps",
           "gt_bg_total_offered_mbps", "gt_n_stas", "gt_n_aps")
HELD_OUT_REGIMES = ("gt_tcp_fraction", "gt_downlink_fraction", "gt_bg_total_offered_mbps")
LABEL_BANDS_MBPS = (0, 1, 10, 30, 50, 61)
TRAIN_SHARES = (0.05, 0.1, 0.25, 0.5, 1.0)

# The rule with nothing fitted: the fastest 802.11n single-stream rate whose
# minimum receiver sensitivity the AP's mean RSSI meets, times the share of
# time the channel was heard idle, times the share of a saturated sender's
# airtime that carries payload.
SENSITIVITY_DBM_AND_RATE_MBPS = ((-82, 6.5), (-79, 13.0), (-77, 19.5), (-74, 26.0),
                                 (-70, 39.0), (-66, 52.0), (-65, 58.5), (-64, 65.0))
PAYLOAD_SHARE = 0.9


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    table = pd.read_csv(args.dataset)
    features = feature_groups(table)
    folds = [split_by_topology(table, test_fold=fold) for fold in range(N_FOLDS)]
    # The validation part is unused here, since nothing is tuned, so it joins
    # the training part.
    folds = [(pd.concat([train, val], ignore_index=True), test) for train, val, test in folds]

    report = {
        "rows": len(table), "scans": int(table.scan_id.nunique()),
        "reference": reference_errors(folds),
        "feature_groups": {name: across_folds(
            [mae(test, fit_predict(train, test, columns)) for train, test in folds])
            for name, columns in features.items()},
        "by_regime": by_regime(folds, features["all"]),
        "held_out_regime": {regime: held_out(table, regime, features["all"])
                            for regime in HELD_OUT_REGIMES},
        "learning_curve": learning_curve(folds, features["all"]),
        "label_factors": label_factors(table, args.runs_dir, features["all"]),
        "privileged": privileged(table, args.runs_dir, features["all"]),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "regime_analysis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


# ---------------------------------------------------------------------------
# The model and its error
#
# One gradient-boosted regressor at one setting serves every part, fitted to
# the absolute error it is scored on. Nothing is tuned, so the parts differ
# only in the rows and columns they hand it.
# ---------------------------------------------------------------------------

def fit_predict(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> np.ndarray:
    model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=400,
                                          learning_rate=0.05, random_state=0)
    model.fit(train[columns], train[LABEL_COL])
    return np.clip(model.predict(test[columns]), 0.0, None)


def mae(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(frame[LABEL_COL].to_numpy() - prediction)))


def across_folds(values: list[float]) -> dict:
    return {"mean": round(float(np.mean(values)), 3), "std": round(float(np.std(values)), 3)}


# ---------------------------------------------------------------------------
# 1. Reference errors
# ---------------------------------------------------------------------------

def reference_errors(folds) -> dict:
    return {
        "training_median": across_folds(
            [mae(test, np.full(len(test), train[LABEL_COL].median())) for train, test in folds]),
        "rate_times_idle_share": across_folds(
            [mae(test, rate_times_idle_share(test)) for _, test in folds]),
    }


def rate_times_idle_share(frame: pd.DataFrame) -> np.ndarray:
    rssi = frame["feat_ap_rssi_mean"].to_numpy()
    rate = np.zeros(len(frame))
    for sensitivity_dbm, rate_mbps in SENSITIVITY_DBM_AND_RATE_MBPS:
        rate[rssi >= sensitivity_dbm] = rate_mbps
    return PAYLOAD_SHARE * rate * (1.0 - frame["feat_chan_busy_frac"].to_numpy())


# ---------------------------------------------------------------------------
# 2. Feature groups
# ---------------------------------------------------------------------------

def feature_groups(table: pd.DataFrame) -> dict[str, list[str]]:
    every = [c for c in table.columns if c.startswith("feat_")]
    rssi = [c for c in every if c.startswith("feat_ap_rssi")]
    channel = [c for c in every if c.startswith("feat_chan_")]
    return {"rssi": rssi, "channel": channel, "rssi_and_channel": rssi + channel, "all": every}


# ---------------------------------------------------------------------------
# 3. Error by regime
# ---------------------------------------------------------------------------

def by_regime(folds, columns: list[str]) -> dict:
    """Pooled over folds, the error within each third of each regime parameter
    and each band of the label, with the label's spread there for scale."""
    pooled = pd.concat([test.assign(prediction=fit_predict(train, test, columns))
                        for train, test in folds], ignore_index=True)
    pooled["label_band"] = pd.cut(pooled[LABEL_COL], LABEL_BANDS_MBPS, right=False)
    report = {}
    for regime in (*REGIMES, "label_band"):
        groups = (pooled[regime] if regime in ("label_band", "gt_n_aps")
                  else pd.qcut(pooled[regime], 3, duplicates="drop"))
        report[regime] = {
            str(value): {"rows": len(part), "mae": round(mae(part, part.prediction), 3),
                         "label_mean": round(float(part[LABEL_COL].mean()), 2),
                         "label_sd": round(float(part[LABEL_COL].std()), 2)}
            for value, part in pooled.groupby(groups, observed=True)}
    return report


# ---------------------------------------------------------------------------
# 4. Held-out regimes
# ---------------------------------------------------------------------------

def held_out(table: pd.DataFrame, regime: str, columns: list[str]) -> dict:
    """Error on one half of a regime parameter's range after fitting on the
    other half, beside a fit on as many topologies from the whole range.

    The test rows are the same for both fits: the test-fold topologies that lie
    in the held-out half, pooled over folds.
    """
    threshold = table.drop_duplicates(TOPOLOGY_COL)[regime].median()
    report = {}
    for name, is_held_out in (("fit_low_test_high", table[regime] > threshold),
                              ("fit_high_test_low", table[regime] <= threshold)):
        other_half, whole_range = [], []
        for fold in range(N_FOLDS):
            train, val, test = split_by_topology(table.assign(is_held_out=is_held_out),
                                                 test_fold=fold)
            train = pd.concat([train, val], ignore_index=True)
            test = test[test.is_held_out]
            seen = train[~train.is_held_out]
            topologies = train[TOPOLOGY_COL].drop_duplicates()
            sample = topologies.sample(seen[TOPOLOGY_COL].nunique(), random_state=fold)
            mixed = train[train[TOPOLOGY_COL].isin(sample)]
            other_half.append(mae(test, fit_predict(seen, test, columns)))
            whole_range.append(mae(test, fit_predict(mixed, test, columns)))
        report[name] = {"threshold": round(float(threshold), 3),
                        "fit_on_other_half": across_folds(other_half),
                        "fit_on_whole_range": across_folds(whole_range)}
    return report


# ---------------------------------------------------------------------------
# 5. Learning curve
# ---------------------------------------------------------------------------

def learning_curve(folds, columns: list[str]) -> dict:
    report = {}
    for share in TRAIN_SHARES:
        errors = []
        for fold, (train, test) in enumerate(folds):
            topologies = train[TOPOLOGY_COL].drop_duplicates()
            kept = topologies.sample(frac=share, random_state=fold)
            errors.append(mae(test, fit_predict(train[train[TOPOLOGY_COL].isin(kept)],
                                                test, columns)))
        report[str(share)] = across_folds(errors)
    return report


# ---------------------------------------------------------------------------
# 6. The factors of the label
#
# A saturated sender's throughput is close to the product of three things the
# simulator counts: the share of time it transmitted, the PHY rate of its data
# frames and the share of them acknowledged. Each is fitted on its own, on the
# rows whose AP the scan's recording joined and where the candidate sent data.
# ---------------------------------------------------------------------------

FACTORS = ("airtime_share", "phy_rate_mbps", "delivery_ratio")


def label_factors(table: pd.DataFrame, runs_dir: Path, columns: list[str]) -> dict:
    """How much of each factor's variance the features explain, and how much
    of the label's log variance each factor carries."""
    rows = []
    for scan_id, scan in table.groupby("scan_id"):
        candidate = json.loads(
            (runs_dir / f"{scan_id}__draw" / "metadata.json").read_text())["candidate"]
        joined = scan[scan.ap_index == candidate["target_ap"]]
        if len(joined) and candidate["data_frames_sent"] > 0:
            rows.append(joined.iloc[[0]].assign(
                airtime_share=candidate["transmit_seconds"] / candidate["observed_seconds"],
                phy_rate_mbps=candidate["mean_phy_rate_mbps"],
                delivery_ratio=candidate["data_frames_acked"] / candidate["data_frames_sent"]))
    sent = pd.concat(rows, ignore_index=True)

    report = {"rows": len(sent)}
    for factor in FACTORS:
        explained = []
        for fold in range(N_FOLDS):
            train, val, test = split_by_topology(sent, test_fold=fold)
            train = pd.concat([train, val], ignore_index=True)
            model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                                  random_state=0)
            model.fit(train[columns], train[factor])
            residual = test[factor] - model.predict(test[columns])
            explained.append(1.0 - residual.var() / test[factor].var())
        report[factor] = {
            "quartiles": [round(float(q), 3) for q in sent[factor].quantile([.25, .5, .75])],
            "variance_explained": across_folds(explained),
            "log_variance": round(float(np.log(sent[factor].clip(lower=1e-3)).var()), 3)}
    return report


# ---------------------------------------------------------------------------
# 7. Privileged facts
#
# The simulator knows the mean power the client receives from the AP, how many
# stations the AP serves and what they offer, and the same totals for the
# other APs on its channel. A model given these beside the features shows how
# much of the remaining error is information the observation did not deliver.
# ---------------------------------------------------------------------------

PRIVILEGED = ("true_rx_dbm", "true_sta_count", "true_offered_mbps",
              "true_cochannel_sta_count", "true_cochannel_offered_mbps")


def privileged(table: pd.DataFrame, runs_dir: Path, columns: list[str]) -> dict:
    facts = []
    for scan_id in table.scan_id.unique():
        aps = json.loads((runs_dir / f"{scan_id}__draw" / "metadata.json").read_text())["aps"]
        for ap in aps:
            others = [o for o in aps if o["channel"] == ap["channel"] and o is not ap]
            facts.append({
                "scan_id": scan_id, "ap_index": ap["index"],
                "true_rx_dbm": ap["candidate_rx_dbm"],
                "true_sta_count": ap["sta_count"],
                "true_offered_mbps": ap["offered_mbps"],
                "true_cochannel_sta_count": sum(o["sta_count"] for o in others),
                "true_cochannel_offered_mbps": sum(o["offered_mbps"] for o in others)})
    known = table.merge(pd.DataFrame(facts), on=["scan_id", "ap_index"], validate="1:1")

    report = {}
    for name, given in (("features", columns), ("privileged", list(PRIVILEGED)),
                        ("features_and_privileged", columns + list(PRIVILEGED))):
        errors = []
        for fold in range(N_FOLDS):
            train, val, test = split_by_topology(known, test_fold=fold)
            errors.append(mae(test, fit_predict(pd.concat([train, val]), test, given)))
        report[name] = across_folds(errors)
    return report


if __name__ == "__main__":
    raise SystemExit(main())
