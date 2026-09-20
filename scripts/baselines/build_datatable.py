#!/usr/bin/env python3
"""This file builds an aggregate table: one row of summary statistics per target AP.

This is to be run once the dataset cache has been constructed by tf/cache_dataset.py. 
This file constructs a single tabular database, in which a row summarises all of one
observation window into the statistics the throughput of one AP is predicted from. 
A row is emitted for every target AP of every scan.

INPUT
  cache_dir, written by tf/cache_dataset.py: <scan_id>.npz per scan, and
  _index.npz.

OUTPUT: a CSV with one row per (scan, valid AP)
  topology_id, scan_id, ap_index   identifiers; ap_index counts the valid APs
  feat_chan_*  the AP's channel
  feat_ap_*    the AP's own BSS
  feat_rel_*   the AP against the other valid APs of its scan
  label_*      throughput after joining the AP
  gt_*         the deployment's AP and hotspot counts, for slicing; never input

PROCESS, per scan
  1. Undo the log1p the cache stored airtime, length and rate under.
  2. Summarise the frames heard on each channel a valid AP occupies.
  3. Summarise each valid AP's own BSS, taking from its window descriptor the
     beacon levels, the airtime fraction and the client count.
  4. Compare each valid AP with the others of its scan.

Run:
  .venv/bin/python3 scripts/baselines/build_datatable.py data/cache \
      --out data/aggregate.csv
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.common.projection import (DWELL_S, LISTEN_PER_DWELL_S,  # noqa: E402
                                       SCAN_CHANNELS, TYPE_CTRL, TYPE_DATA, TYPE_MGMT)


def scan_rows(path: Path, scan_id: str, topology_id: str, ground_truth: dict,
              names: dict[str, list[str]]) -> list[dict]:
    """The rows of one cached scan, one per valid AP."""
    with np.load(path) as file:
        cached = {key: file[key] for key in file.files}
    scan = CachedScan(cached, names)

    rows = []
    for number, channel in enumerate(cached["ap_channels"]):
        rows.append({
            "topology_id": topology_id,
            "scan_id": scan_id,
            "ap_index": number,
            **scan.channel_features(int(channel)),
            **scan.ap_features(number),
            **scan.relative_features(number),
            "label_throughput_mbps": float(cached["labels"][number]),
            **ground_truth,
        })
    return rows


class CachedScan:
    """One scan's frames and descriptors, with the log1p columns undone.

    Every rate and fraction divides by the time the radio spent receiving on one
    channel, as the older builder did: the same for all four, since the sweep
    gives each the same number of dwells.
    """

    def __init__(self, cached: dict[str, np.ndarray], names: dict[str, list[str]]):
        frame = {name: cached["frames"][:, i] for i, name in enumerate(names["frames"])}
        self.category = cached["frame_categories"]
        self.ap_number = cached["frame_ap"]
        self.rssi = frame["rssi_dbm"]
        self.airtime_us = np.expm1(frame["airtime_log1p"])
        self.length = np.expm1(frame["length_log1p"])
        self.rate = np.expm1(frame["rate_log1p"])
        self.beacon = frame["is_beacon"] > 0.5
        self.retry = frame["is_retry"] > 0.5
        # dwells run back to back, so a dwell index and a position within it
        # place a frame in the window up to the constant the sweep started at
        self.time_s = (cached["frame_dwell"] + frame["offset_in_dwell"]) * DWELL_S

        self.dwell_channel = cached["dwell_channel"]
        self.frame_channel = self.dwell_channel[cached["frame_dwell"]]
        self.cca = cached["cca_samples"]
        self.listen_s = len(self.dwell_channel) // len(SCAN_CHANNELS) * LISTEN_PER_DWELL_S
        self.descriptors = {name: cached["descriptors"][:, i]
                            for i, name in enumerate(names["descriptors"])}
        self.n_valid_aps = len(cached["ap_channels"])

    def channel_features(self, channel: int) -> dict:
        """feat_chan_*: everything heard on one channel over the window."""
        on = self.frame_channel == channel
        data = on & (self.category == TYPE_DATA)
        heard = int(on.sum())
        return {
            "feat_chan_busy_frac": float(self.airtime_us[on].sum()) / (self.listen_s * 1e6),
            "feat_chan_cca_busy_frac": float(self.cca[self.dwell_channel == channel].mean()),
            "feat_chan_frames_per_s": heard / self.listen_s,
            "feat_chan_bytes_per_s": float(self.length[on].sum()) / self.listen_s,
            "feat_chan_data_frac": float(data.sum()) / heard if heard else 0.0,
            "feat_chan_mgmt_frac": (float((on & (self.category == TYPE_MGMT)).sum()) / heard
                                    if heard else 0.0),
            "feat_chan_ctrl_frac": (float((on & (self.category == TYPE_CTRL)).sum()) / heard
                                    if heard else 0.0),
            "feat_chan_retry_frac": float((on & self.retry).sum()) / heard if heard else 0.0,
            "feat_chan_mean_data_rate": mean(self.rate[data]),
            "feat_chan_mean_rssi": mean(self.rssi[on]),
        }

    def ap_features(self, number: int) -> dict:
        """feat_ap_*: one valid AP's own BSS over the window.

        The beacon levels, the airtime fraction and the client count are the
        window descriptor's, which is what the cell models read; the rest come
        from the AP's frames. A client count needs the addresses the cache does
        not keep, so nothing else can supply it.
        """
        descriptor = {name: float(values[number])
                      for name, values in self.descriptors.items()}
        mine = self.ap_number == number
        data = mine & (self.category == TYPE_DATA)
        heard = int(mine.sum())
        gaps = np.diff(np.sort(self.time_s[mine & self.beacon]))
        return {
            "feat_ap_rssi_mean": descriptor["beacon_rssi_mean"],
            "feat_ap_rssi_std": descriptor["beacon_rssi_std"],
            "feat_ap_rssi_max": descriptor["beacon_rssi_max"],
            "feat_ap_rssi_min": descriptor["beacon_rssi_min"],
            "feat_ap_rssi_last": descriptor["beacon_rssi_last"],
            "feat_ap_beacons": round(np.expm1(descriptor["beacons_log1p"])),
            # the spacing of DECODED beacons, which the sweep's revisits set
            # rather than the AP's beacon interval; two beacons make one gap
            "feat_ap_beacon_gap_mean": mean(gaps),
            "feat_ap_beacon_gap_std": spread(gaps),
            "feat_ap_beacon_gap_known": float(len(gaps) > 0),
            "feat_ap_frames_per_s": heard / self.listen_s,
            "feat_ap_bytes_per_s": float(self.length[mine].sum()) / self.listen_s,
            "feat_ap_airtime_frac": descriptor["bss_airtime_fraction"],
            "feat_ap_n_clients": round(np.expm1(descriptor["bss_transmitters_log1p"])),
            "feat_ap_data_frames": int(data.sum()),
            "feat_ap_mean_data_rate": mean(self.rate[data]),
            "feat_ap_retry_frac": float((mine & self.retry).sum()) / heard if heard else 0.0,
        }

    def relative_features(self, number: int) -> dict:
        """feat_rel_*: one valid AP against the others of its scan.

        The airtime share divides by the airtime of every BSS heard, which is
        every frame that names one: only control frames name none. The client
        share divides by the scan's valid APs alone, since the clients of a BSS
        whose beacons never decoded are not in the cache.
        """
        levels = self.descriptors["beacon_rssi_mean"]
        mine = float(levels[number])
        others = np.delete(levels, number)
        named_bss = self.category != TYPE_CTRL
        all_airtime = float(self.airtime_us[named_bss].sum()) / (self.listen_s * 1e6)
        clients = np.round(np.expm1(self.descriptors["bss_transmitters_log1p"]))
        return {
            "feat_rel_n_options": float(self.n_valid_aps),
            # options STRICTLY stronger, so options tied in whole dBm share a rank
            "feat_rel_rssi_rank": float((levels > mine).sum()),
            "feat_rel_rssi_margin_best_other": mine - float(others.max()) if len(others) else 0.0,
            "feat_rel_rssi_minus_mean": mine - float(levels.mean()),
            "feat_rel_airtime_share": (float(self.descriptors["bss_airtime_fraction"][number])
                                       / (all_airtime or 1.0)),
            "feat_rel_clients_share": (float(clients[number]) / float(clients.sum())
                                       if clients.sum() else 0.0),
        }


def mean(values: np.ndarray) -> float:
    """The mean, or NaN where there is nothing to average, as the tabular models
    read missing features: impute_features decides what to fill them with."""
    return float(values.mean()) if len(values) else float("nan")


def spread(values: np.ndarray) -> float:
    """The population standard deviation; 0.0 for a single value, NaN for none."""
    if len(values) > 1:
        return float(statistics.pstdev(values.tolist()))
    return 0.0 if len(values) else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/aggregate.csv"))
    args = parser.parse_args()

    index = np.load(args.cache_dir / "_index.npz")
    names = {"frames": index["frame_features"].tolist(),
             "descriptors": index["descriptor_features"].tolist()}
    rows = []
    for i, scan_id in enumerate(index["scan_ids"], 1):
        rows.extend(scan_rows(
            args.cache_dir / f"{scan_id}.npz", str(scan_id), str(index["topology_ids"][i - 1]),
            {"gt_n_aps": int(index["configured_n_aps"][i - 1]),
             "gt_n_hotspots": int(index["n_hotspots"][i - 1])}, names))
        if i % 500 == 0 or i == len(index["scan_ids"]):
            print(f"[{i}/{len(index['scan_ids'])}] {len(rows)} rows", flush=True)

    # imported here rather than at module scope: only writing needs it
    import pandas as pd

    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"wrote {len(table)} rows x {len(table.columns)} cols "
          f"({table.scan_id.nunique()} scans) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
