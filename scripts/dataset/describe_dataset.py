#!/usr/bin/env python3
"""Random regret and discovery of an aggregate table.

Reads an aggregate table from build_aggregate_table.py. The binned and frame
corpora have the same scans, options and labels, so the numbers hold for them
too. Both tables give one row per deployment size, then one over all scans.

  RANDOM REGRET  If the client joins one of the APs it heard at random, how much
                 throughput does it lose against joining the best one? Per
                 scan, the oracle gets the highest throughput among the options,
                 a random pick the mean, and the worst pick the lowest. Random
                 regret is oracle minus random. Values are means over scans, in
                 Mbit/s.

  DISCOVERY      How much of the deployment does passive scanning find? aps is
                 the share of the deployment's APs whose beacon decoded.
                 stations is the share of the heard APs' stations that sent a
                 decoded frame (feat_ap_n_clients), against the true count
                 from metadata.json. Values are means over scans, in percent.

Run:  python scripts/dataset/scan_stats.py data/aggregate.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.models.data import load_dataset  # noqa: E402


def by_ap_count(per_scan: pd.DataFrame, columns: dict) -> pd.DataFrame:
    """columns aggregated per deployment AP count, then over every scan."""
    table = per_scan.groupby("n_aps").agg(**columns)
    table.index = [f"{n} APs" for n in table.index]
    total = per_scan.assign(group="all scans").groupby("group").agg(**columns)
    return pd.concat([table, total])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("table", type=Path)
    args = parser.parse_args()

    df = load_dataset(args.table)
    per_scan = df.groupby("scan_id").agg(
        n_aps=("gt_n_aps", "first"),
        n_options=("ap_index", "size"),
        oracle=("label_throughput_mbps", "max"),
        random=("label_throughput_mbps", "mean"),
        worst=("label_throughput_mbps", "min"),
        stations_heard=("feat_ap_n_clients", "sum"),
        stations=("gt_ap_sta_count", "sum"),
    )
    per_scan["regret"] = per_scan.oracle - per_scan.random
    per_scan["aps_found"] = 100 * per_scan.n_options / per_scan.n_aps
    per_scan["stations_found"] = 100 * per_scan.stations_heard / per_scan.stations

    print(f"{len(per_scan)} scans from {df.topology_id.nunique()} topologies, "
          f"{len(df)} options")

    print("\n=== RANDOM REGRET (mean Mbit/s per scan) ===")
    print(by_ap_count(per_scan, dict(
        scans=("regret", "size"),
        oracle=("oracle", "mean"),
        random=("random", "mean"),
        worst=("worst", "mean"),
        random_regret=("regret", "mean"),
    )).round(2).to_string())

    print("\n=== DISCOVERY (mean % per scan) ===")
    print(by_ap_count(per_scan, dict(
        aps=("aps_found", "mean"),
        stations=("stations_found", "mean"),
    )).round(1).to_string())


if __name__ == "__main__":
    main()
