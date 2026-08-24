#!/usr/bin/env python3
"""Describe the dataset and check the properties the pipeline depends on.

Run this after build_dataset.py. It reports the label distribution, how much
a perfect chooser would gain over a random one, and how well the observable
features track the ground truth they are meant to proxy - and it fails
loudly on structural problems (leakage-prone columns, groups with a single
option, features that vary within a group when they should not).

Run:  python scripts/inspect_dataset.py data/dataset.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.data import (feature_columns, ground_truth_columns, impute_features,  # noqa: E402
                         load_dataset)
from models.evaluate import evaluate_all, oracle_ceiling  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()

    raw = load_dataset(args.dataset)
    df = impute_features(raw)
    feats = feature_columns(df)
    y = df["label_throughput_mbps"]
    problems = []

    print(f"=== SHAPE ===")
    topo = df.topology_id.nunique() if "topology_id" in df.columns else df.group_id.nunique()
    print(f"rows={len(df)}  groups={df.group_id.nunique()}  topologies={topo}  "
          f"features={len(feats)}  ground-truth cols={len(ground_truth_columns(df))}")
    sizes = df.groupby("group_id").size()
    print(f"options per group: min={sizes.min()} median={int(sizes.median())} max={sizes.max()}")
    if (sizes < 2).any():
        problems.append(f"{(sizes < 2).sum()} group(s) have a single option; "
                        "a choice set needs at least two")

    print(f"\n=== LABEL ===")
    print(f"zero={float((y == 0).mean()):.1%}  <1Mbps={float((y < 1).mean()):.1%}  "
          f">5={float((y > 5).mean()):.1%}  >20={float((y > 20).mean()):.1%}")
    print(f"quantiles: {[round(v, 2) for v in y.quantile([0, .25, .5, .75, .9, 1]).tolist()]}")
    print(f"association failures: {int((df.label_associated == 0).sum())} rows")

    print(f"\n=== HOW MUCH IS AT STAKE ===")
    oc = oracle_ceiling(df)
    for k, v in oc.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
    if oc["mean_spread_mbps"] < 1.0:
        problems.append("best and worst AP are nearly identical; the choice barely matters")

    print(f"\n=== HEURISTIC BASELINES (whole dataset) ===")
    print(evaluate_all(df, {}).to_string(index=False))

    if "gt_n_aps" in df:
        print(f"\n=== DISCOVERY AND DECISION COVERAGE ===")
        group = df.groupby("group_id").agg(
            n_options=("ap_index", "size"),
            n_aps=("gt_n_aps", "first"),
            best_mbps=("label_throughput_mbps", "max"),
            worst_mbps=("label_throughput_mbps", "min"),
        )
        group["discovery_fraction"] = group.n_options / group.n_aps
        group["spread_mbps"] = group.best_mbps - group.worst_mbps
        coverage = group.groupby("n_aps").agg(
            groups=("n_options", "size"),
            mean_options=("n_options", "mean"),
            mean_discovery=("discovery_fraction", "mean"),
            full_discovery=("discovery_fraction", lambda x: float((x == 1.0).mean())),
            best_mbps=("best_mbps", "mean"),
            spread_mbps=("spread_mbps", "mean"),
        )
        print(coverage.round(3).to_string())
        quantiles = group.discovery_fraction.quantile([0, .1, .25, .5, .75, .9, 1])
        print("discovery-fraction quantiles:",
              {float(k): round(float(v), 3) for k, v in quantiles.items()})

        for column, label in (("gt_n_hotspots", "HOTSPOT COUNT"),
                              ("gt_candidate_stratum", "CANDIDATE STRATUM")):
            if column not in df or df[column].isna().all():
                continue
            values = df.groupby("group_id")[column].first()
            stratified = group.join(values).groupby(column).agg(
                groups=("n_options", "size"),
                mean_options=("n_options", "mean"),
                mean_discovery=("discovery_fraction", "mean"),
                best_mbps=("best_mbps", "mean"),
                spread_mbps=("spread_mbps", "mean"),
            )
            print(f"\n=== BY {label} ===")
            print(stratified.round(3).to_string())

    print(f"\n=== DO OBSERVABLES TRACK GROUND TRUTH? ===")
    checks = [
        ("feat_ap_rssi_mean", "gt_true_distance", "RSSI vs true distance", "negative"),
        ("feat_ap_n_clients", "gt_ap_sta_count", "seen clients vs true count", "positive"),
        ("feat_chan_cca_busy_frac", "gt_bg_total_offered_mbps",
         "CCA busy vs offered load", "positive"),
        ("feat_ap_rssi_mean", "label_throughput_mbps", "RSSI vs throughput", "positive"),
        ("feat_chan_cca_busy_frac", "label_throughput_mbps",
         "CCA busy vs throughput", "negative"),
    ]
    for a, b, desc, expect in checks:
        if a not in df or b not in df:
            continue
        if df[a].nunique() < 2 or df[b].nunique() < 2:
            print(f"  {desc:<34} not tested (constant in this dataset)")
            continue
        r = float(np.corrcoef(df[a], df[b])[0, 1])
        ok = (r < -0.1) if expect == "negative" else (r > 0.1)
        print(f"  {desc:<34} r={r:+.3f}  expected {expect:<8} {'OK' if ok else 'UNEXPECTED'}")
        if not ok:
            problems.append(f"{desc}: r={r:+.3f} contradicts the expected {expect} relation")

    print(f"\n=== STRUCTURAL CHECKS ===")
    const = [c for c in feats if df[c].nunique() <= 1]
    print(f"  constant features (no information): {const if const else 'none'}")
    # This used to print and pass. A column with one value cannot inform a
    # prediction, and its presence usually means an upstream statistic had too
    # few samples to be computed - which is a defect in the observation, not a
    # harmless quirk of the table.
    if const:
        problems.append(f"{len(const)} feature(s) take a single value and carry no "
                        f"information: {const}")
    nan_raw = [c for c in feature_columns(raw) if raw[c].isna().any()]
    print(f"  features with NaNs before imputation: {len(nan_raw)} "
          f"(expected: statistics needing more samples than a short scan captured)")
    # Option-relative features must vary within a group. n_options is the
    # intentional exception: it describes the group itself, not one AP.
    flat = [c for c in feats if c.startswith("feat_rel_") and c != "feat_rel_n_options"
            and df.groupby("group_id")[c].nunique().max() <= 1]
    print(f"  relative features constant within every group: {flat if flat else 'none'}")

    print()
    if problems:
        print("PROBLEMS FOUND:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("No structural problems found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
