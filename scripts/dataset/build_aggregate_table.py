#!/usr/bin/env python3
"""Build the aggregate table: each option's frames summarised over the whole window.

See README.md in this directory for how the table relates to the other builders.

INPUT
  runs_dir  the run directories and _manifest.json written by run_sweep.py

PROCESS, per scan
  1. Project the scan's recording onto one radio (projection.project) and find
     its options (scans.find_options).
  2. Summarise the kept frames per channel, per BSS, and per option against the
     scan's other options.
  3. Join each option to its run's label and simulator parameters.

OUTPUT: a CSV with one row per (scan, option)
  topology_id, scan_id, ap_index, run_id   identifiers
  feat_chan_*  the option's channel
  feat_ap_*    the option's own BSS
  feat_rel_*   the option against the other options of its scan
  label_*      throughput and association after joining the option
  gt_*         simulator parameters; never model input
  meta_*       the window, whether it was projected, and the listening time
"""

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.dataset.projection import (SCAN_CHANNELS, TYPE_CTRL, TYPE_DATA,  # noqa: E402
                                        TYPE_MGMT, describe, project)
from scripts.dataset.scans import Scan, find_options, find_scans  # noqa: E402


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def _std(xs):
    xs = [x for x in xs if x is not None]
    return statistics.pstdev(xs) if len(xs) > 1 else (0.0 if xs else None)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def busy_fraction(busy: list[dict]) -> dict[int, float]:
    """The mean busy share of each scan channel over the projected chanbusy.csv
    milliseconds on it; 0.0 for a channel with none, since the simulator writes
    rows only for occupied channels."""
    shares: dict[int, list[float]] = defaultdict(list)
    for e in busy:
        shares[e["channel"]].append(e["busy_frac"])
    return {ch: statistics.fmean(shares[ch]) if shares[ch] else 0.0 for ch in SCAN_CHANNELS}


def channel_features(frames: list[dict], listen_s: float,
                     busy_frac: float) -> tuple[dict, dict[str, dict]]:
    """feat_chan_* for one channel, and feat_ap_* for every BSSID heard on it.

    frames are the projected frames on this channel, and listen_s the time the
    radio spent receiving on it; every rate and fraction is divided by listen_s.
    """
    total_airtime_us = sum(r["dur"] for r in frames)
    data_rows = [r for r in frames if r["cat"] == TYPE_DATA]

    chan = {
        "feat_chan_busy_frac": total_airtime_us / (listen_s * 1e6) if listen_s else 0.0,
        "feat_chan_cca_busy_frac": busy_frac,
        "feat_chan_frames_per_s": len(frames) / listen_s if listen_s else 0.0,
        "feat_chan_bytes_per_s": sum(r["len"] for r in frames) / listen_s if listen_s else 0.0,
        "feat_chan_data_frac": len(data_rows) / len(frames) if frames else 0.0,
        "feat_chan_mgmt_frac": sum(1 for r in frames if r["cat"] == TYPE_MGMT) / len(frames) if frames else 0.0,
        "feat_chan_ctrl_frac": sum(1 for r in frames if r["cat"] == TYPE_CTRL) / len(frames) if frames else 0.0,
        "feat_chan_retry_frac": sum(1 for r in frames if r["retry"]) / len(frames) if frames else 0.0,
        "feat_chan_mean_data_rate": _mean([r["rate"] for r in data_rows]),
        "feat_chan_n_bssids": len({r["bssid"] for r in frames if r["bssid"]}),
        "feat_chan_n_tas": len({r["ta"] for r in frames if r["ta"]}),
        "feat_chan_mean_rssi": _mean([r["rssi"] for r in frames]),
    }

    by_bssid: dict[str, list[dict]] = defaultdict(list)
    for r in frames:
        if r["bssid"]:
            by_bssid[r["bssid"]].append(r)

    per_ap: dict[str, dict] = {}
    for bssid in sorted(by_bssid):
        bss = by_bssid[bssid]
        beacons = sorted((f for f in bss if f["beacon"]), key=lambda f: f["tx_start"])
        beacon_rssi = [f["rssi"] for f in beacons]
        btimes = [f["tx_start"] for f in beacons]
        gaps = [b - a for a, b in zip(btimes, btimes[1:])]
        # transmitters in the BSS other than the AP: the station-count estimate
        clients = {f["ta"] for f in bss if f["ta"] and f["ta"] != bssid}
        dframes = [f for f in bss if f["cat"] == TYPE_DATA]

        per_ap[bssid] = {
            "feat_ap_rssi_mean": _mean(beacon_rssi),
            "feat_ap_rssi_std": _std(beacon_rssi),
            "feat_ap_rssi_max": max(beacon_rssi) if beacon_rssi else None,
            "feat_ap_rssi_min": min(beacon_rssi) if beacon_rssi else None,
            "feat_ap_rssi_last": beacon_rssi[-1] if beacon_rssi else None,
            "feat_ap_beacons": len(beacons),
            # The spacing of DECODED beacons, which the sweep's revisits set
            # rather than the AP's beacon interval. It needs two beacons: with
            # fewer it is None, scripts/models/data.py fills 0.0, and _known
            # marks which rows those are.
            "feat_ap_beacon_gap_mean": _mean(gaps),
            "feat_ap_beacon_gap_std": _std(gaps),
            "feat_ap_beacon_gap_known": float(len(gaps) > 0),
            "feat_ap_frames_per_s": len(bss) / listen_s if listen_s else 0.0,
            "feat_ap_bytes_per_s": sum(f["len"] for f in bss) / listen_s if listen_s else 0.0,
            "feat_ap_airtime_frac": sum(f["dur"] for f in bss) / (listen_s * 1e6) if listen_s else 0.0,
            "feat_ap_n_clients": len(clients),
            "feat_ap_data_frames": len(dframes),
            "feat_ap_mean_data_rate": _mean([f["rate"] for f in dframes]),
            "feat_ap_retry_frac": sum(1 for f in bss if f["retry"]) / len(bss) if bss else 0.0,
        }
    return chan, per_ap


def relative_features(mac: str, rssi_of: dict[str, float],
                      per_ap: dict[str, dict]) -> dict:
    """feat_rel_* for one option against the other options of its scan.

    rssi_of holds the beacon RSSI of every option. The airtime and transmitter
    shares divide by every BSSID heard, options or not, so across a scan's
    options they add up to 1.0 only when every AP that transmitted was also
    discovered.
    """
    mine = rssi_of[mac]
    best_other = max(v for k, v in rssi_of.items() if k != mac)
    # divided by listening time before summing, since channels under a sweep
    # need not have been listened to for equally long
    total_airtime = sum(v["feat_ap_airtime_frac"] for v in per_ap.values()) or 1.0
    total_clients = sum(v["feat_ap_n_clients"] for v in per_ap.values())
    return {
        "feat_rel_n_options": len(rssi_of),
        # options STRICTLY stronger, so options tied in whole dBm share a rank
        "feat_rel_rssi_rank": sum(1 for v in rssi_of.values() if v > mine),
        "feat_rel_rssi_margin_best_other": mine - best_other,
        "feat_rel_rssi_minus_mean": mine - statistics.fmean(rssi_of.values()),
        "feat_rel_airtime_share": per_ap[mac]["feat_ap_airtime_frac"] / total_airtime,
        "feat_rel_clients_share": (per_ap[mac]["feat_ap_n_clients"] / total_clients
                                   if total_clients else 0.0),
    }


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

def provenance_columns(window: float, listen_s: float, all_channels: bool) -> dict:
    """meta_*: the window, whether the recording was projected onto one radio, and
    how long the option's channel was listened to."""
    return {
        "meta_window_s": window,
        "meta_scan_sweep": int(not all_channels),
        "meta_scan_seconds": listen_s,
    }


def ground_truth_columns(meta: dict, target_ap: int) -> dict:
    """gt_*: the run's simulator parameters and what the simulator knows about
    the joined AP."""
    params = meta["params"]
    ap = next(a for a in meta["aps"] if a["index"] == target_ap)
    candidate = meta["candidate"]
    return {
        "gt_rng_seed": meta["rng_seed"],
        "gt_topology_seed": params["topology_seed"],
        "gt_n_aps": params["n_aps"],
        "gt_n_stas": params["n_stas"],
        "gt_ap_spacing": params["ap_spacing"],
        "gt_bg_mean_per_sta_mbps": params["bg_mean_per_sta_mbps"],
        "gt_bg_total_offered_mbps": params["bg_total_offered_mbps"],
        "gt_packet_size": params["packet_size"],
        "gt_n_hotspots": params["n_hotspots"],
        "gt_hotspot_aps": ",".join(map(str, params["hotspot_aps"])),
        "gt_candidate_x": meta["candidate_position"]["x"],
        "gt_candidate_y": meta["candidate_position"]["y"],
        "gt_true_distance": ap["candidate_distance"],
        "gt_ap_sta_count": ap["sta_count"],
        "gt_ap_offered_mbps": ap["offered_mbps"],
        "gt_assoc_delay": candidate["assoc_delay"],
        "gt_observed_seconds": candidate["observed_seconds"],
        "gt_n_channels": params["n_channels"],
        "gt_ap_channel": int(ap["channel"]),
    }


def build_scan_rows(scan: Scan, all_channels: bool) -> list[dict]:
    """The rows of one scan, one per option.

    The features come from the scan's one recording. Each option adds the label
    of the run that joined it, and the gt_* columns from that run's metadata.
    """
    projected = project(scan.recording, scan.scan_id, all_channels)
    options = find_options(scan, projected)
    if options is None:
        return []
    busy = busy_fraction(projected.busy)
    frames_on: dict[int, list[dict]] = defaultdict(list)
    for r in projected.frames:
        frames_on[r["channel"]].append(r)

    chan_feats: dict[int, dict] = {}
    per_ap: dict[str, dict] = {}
    for ch in SCAN_CHANNELS:
        chan_feats[ch], ap_feats = channel_features(
            frames_on[ch], projected.listen_s, busy[ch])
        per_ap.update(ap_feats)
    rssi_of = {option.mac: per_ap[option.mac]["feat_ap_rssi_mean"] for option in options}

    rows = []
    for option in options:
        rows.append({
            "topology_id": scan.topology_id,
            "scan_id": scan.scan_id,
            "ap_index": option.index,
            "run_id": option.metadata["run_id"],
            **provenance_columns(projected.window, projected.listen_s, all_channels),
            **chan_feats[option.channel],
            **per_ap[option.mac],
            **relative_features(option.mac, rssi_of, per_ap),
            "label_throughput_mbps": option.throughput_mbps,
            "label_associated": int(option.associated),
            **ground_truth_columns(option.metadata, option.index),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/aggregate.csv"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--all-channels", action="store_true",
                        help="keep every channel for the whole window instead of "
                             "projecting onto one radio")
    args = parser.parse_args()

    scans = find_scans(args.runs_dir)
    print(f"{len(scans)} scans under {args.runs_dir}")
    print(f"projection: {describe(args.all_channels)}")

    build = functools.partial(build_scan_rows, all_channels=args.all_channels)
    rows: list[dict] = []
    if args.workers <= 1:
        iterator = map(build, scans)
    else:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=args.workers)
        iterator = pool.map(build, scans, chunksize=16)
    try:
        for i, scan_rows in enumerate(iterator, 1):
            rows.extend(scan_rows)
            if i % 200 == 0 or i == len(scans):
                print(f"[{i}/{len(scans)}] {len(rows)} rows")
    finally:
        if args.workers > 1:
            pool.shutdown()

    if not rows:
        sys.exit("no rows extracted")

    # imported here, not at module scope: the worker processes never need it
    import pandas as pd

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows x {len(df.columns)} cols "
          f"({df.scan_id.nunique()} scans) to {args.out}")


if __name__ == "__main__":
    main()
