"""Per-station airtime features for every (scan, discovered AP) row (scratch driver).

INPUT   runs_dir of simulator runs and the aggregate table built from its cache.
OUTPUT  the aggregate table with feat_sta_* columns appended: how the airtime
        heard on the AP's channel divides among individual transmitters, and
        the share an extra saturated station would get under equal sharing.

Run:
  .venv/bin/python3 scripts/scratch_v2/station_features.py data/runs_v2 \
      data/aggregate_v2.csv data/aggregate_v2_stations.csv 4
"""
import concurrent.futures
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.common.parse_scans import find_discovered_aps, find_scans  # noqa: E402
from scripts.common.projection import LISTEN_PER_DWELL_S, project  # noqa: E402

TOP_TRANSMITTERS = 5


def main():
    runs_dir, table_path, out_path, workers = sys.argv[1:5]
    with concurrent.futures.ProcessPoolExecutor(int(workers)) as pool:
        rows = [row for rows in pool.map(scan_rows, find_scans(Path(runs_dir)), chunksize=16)
                for row in rows]
    table = pd.read_csv(table_path).merge(pd.DataFrame(rows), on=["scan_id", "ap_index"],
                                          validate="1:1")
    table.to_csv(out_path, index=False)
    print(f"wrote {len(table)} rows x {table.shape[1]} cols to {out_path}")


def scan_rows(scan) -> list[dict]:
    """The feat_sta_* columns of each discovered AP of one scan."""
    projected = project(scan.recording, scan.scan_id, all_channels=False)
    discovered_aps = find_discovered_aps(scan, projected)
    if discovered_aps is None:
        return []
    rows = []
    for number, ap in enumerate(discovered_aps):
        listen_us = 1e6 * LISTEN_PER_DWELL_S * sum(c == ap.channel
                                                   for c in projected.dwell_schedule)
        airtime, own_bss = defaultdict(float), defaultdict(float)
        unattributed = 0.0
        for frame in projected.frames:
            if frame["channel"] != ap.channel:
                continue
            if frame["ta"] is None:
                unattributed += frame["dur"] / listen_us
                continue
            airtime[frame["ta"]] += frame["dur"] / listen_us
            if frame["bssid"] == ap.mac:
                own_bss[frame["ta"]] += frame["dur"] / listen_us
        shares = np.sort(np.array(list(airtime.values())))[::-1]
        own_stations = np.sort(np.array([a for ta, a in own_bss.items() if ta != ap.mac]))[::-1]
        top = np.zeros(TOP_TRANSMITTERS)
        top[:len(shares[:TOP_TRANSMITTERS])] = shares[:TOP_TRANSMITTERS]
        rows.append({
            "scan_id": scan.scan_id, "ap_index": number,
            "feat_sta_transmitters": float(len(shares)),
            **{f"feat_sta_airtime_top{i + 1}": top[i] for i in range(TOP_TRANSMITTERS)},
            "feat_sta_airtime_rest": float(shares[TOP_TRANSMITTERS:].sum()),
            "feat_sta_unattributed_airtime": unattributed,
            "feat_sta_above_5pct": float((shares > 0.05).sum()),
            "feat_sta_own_ap_airtime": own_bss.get(ap.mac, 0.0),
            "feat_sta_own_station_top1": float(own_stations[0]) if len(own_stations) else 0.0,
            "feat_sta_own_stations_airtime": float(own_stations.sum()),
            "feat_sta_equal_share": equal_share(shares, 1.0 - unattributed),
        })
    return rows


def equal_share(demands: np.ndarray, available: float) -> float:
    """The airtime level L a saturated newcomer reaches when every transmitter
    keeps min(its demand, L) and the newcomer takes L: L + sum(min(d, L)) = available."""
    low, high = 0.0, max(available, 0.0)
    for _ in range(40):
        level = (low + high) / 2
        if level + np.minimum(demands, level).sum() > available:
            high = level
        else:
            low = level
    return low


if __name__ == "__main__":
    main()
