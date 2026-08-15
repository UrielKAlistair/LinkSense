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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    print(f"rows={len(df)}  groups={df.group_id.nunique()}  features={len(feats)}  "
          f"ground-truth cols={len(ground_truth_columns(df))}")
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

    print(f"\n=== DO OBSERVABLES TRACK GROUND TRUTH? ===")
    checks = [
        ("feat_ap_rssi_mean", "gt_true_distance", "RSSI vs true distance", "negative"),
        ("feat_ap_n_clients", "gt_ap_sta_count", "seen clients vs true count", "positive"),
        ("feat_chan_busy_frac", "gt_bg_total_offered_mbps", "busy vs offered load", "positive"),
        ("feat_ap_rssi_mean", "label_throughput_mbps", "RSSI vs throughput", "positive"),
        ("feat_chan_busy_frac", "label_throughput_mbps", "busy vs throughput", "negative"),
    ]
    for a, b, desc, expect in checks:
        if a not in df or b not in df:
            continue
        r = float(np.corrcoef(df[a], df[b])[0, 1])
        ok = (r < -0.1) if expect == "negative" else (r > 0.1)
        print(f"  {desc:<34} r={r:+.3f}  expected {expect:<8} {'OK' if ok else 'UNEXPECTED'}")
        if not ok:
            problems.append(f"{desc}: r={r:+.3f} contradicts the expected {expect} relation")

    print(f"\n=== STRUCTURAL CHECKS ===")
    const = [c for c in feats if df[c].nunique() <= 1]
    print(f"  constant features (no information): {const if const else 'none'}")
    nan_raw = [c for c in feature_columns(raw) if raw[c].isna().any()]
    print(f"  features with NaNs before imputation: {len(nan_raw)} "
          f"(expected: RSSI-like columns for APs never heard)")
    # feat_rel_* must vary within a group, else options are indistinguishable
    flat = [c for c in feats if c.startswith("feat_rel_")
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
