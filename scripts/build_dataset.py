#!/usr/bin/env python3
"""Turn a directory of ns-3 my-wifi-test runs into one flat training table.

For each run (a directory containing metadata.json and a candidate-side
pcap), this extracts features from ONLY the pre-association window
[0, candidate_start_time) - i.e. only signal the candidate's radio could
plausibly have observed before deciding to join an AP - and pairs them with
the post-association throughput label recorded in metadata.json.

The pcap is captured in promiscuous mode on the candidate's own device, so
even before it associates with anything it overhears beacons (and, for any
AP within earshot, ordinary data/control traffic) from every nearby BSS -
that's what makes pre-association per-AP features possible at all.

Column naming convention (enforced, not just a suggestion - see
FEATURE_COLUMNS/GROUND_TRUTH_COLUMNS below and feature_columns() in
models/data.py):
  - "feat_*"  - derived purely from the pre-association pcap window; these
                are the only columns a model should ever train on.
  - "gt_*"    - simulator ground truth / scenario config (true distance,
                true background load, RNG seed, ...). Useful for analysis,
                sanity-checking the feat_* proxies, and stratifying eval -
                NEVER a model input, since a real STA cannot observe these
                before joining.
  - "label_*" - the thing being predicted.
  - everything else (run_id, ...) is bookkeeping.
"""

import argparse
import concurrent.futures
import json
import statistics
import subprocess
import sys
from pathlib import Path

TSHARK_FIELDS = [
    "frame.time_relative",
    "frame.len",
    "wlan.fc.type",
    "wlan.fc.type_subtype",
    "wlan.bssid",
    "radiotap.dbm_antsignal",
]

BEACON_SUBTYPE = "0x0008"


def run_tshark(pcap: Path, end_time: float) -> list[dict]:
    display_filter = f"frame.time_relative < {end_time}"
    cmd = ["tshark", "-r", str(pcap), "-Y", display_filter, "-T", "fields"]
    for f in TSHARK_FIELDS:
        cmd += ["-e", f]

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    rows = []
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        while len(parts) < len(TSHARK_FIELDS):
            parts.append("")
        time_s, frame_len, wlan_type, wlan_subtype, bssid, rssi = parts[: len(TSHARK_FIELDS)]
        rows.append({
            "time": float(time_s) if time_s else 0.0,
            "len": int(frame_len) if frame_len else 0,
            "type": wlan_type,
            "subtype": wlan_subtype,
            "bssid": bssid or None,
            "rssi": float(rssi) if rssi else None,
        })
    return rows


def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def safe_std(xs):
    xs = [x for x in xs if x is not None]
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0 if xs else None


def extract_features(pcap: Path, meta: dict) -> dict:
    candidate_start = meta["params"]["candidate_start_time"]
    rows = run_tshark(pcap, candidate_start)

    aps = meta["aps"]
    target_ap_index = meta["candidate"]["target_ap"]
    target_ap = next(ap for ap in aps if ap["index"] == target_ap_index)
    target_mac = target_ap["mac"].lower()

    # --- global (whole-channel) pre-association features ---
    packet_count = len(rows)
    byte_count = sum(r["len"] for r in rows)
    mgmt_count = sum(1 for r in rows if r["type"] == "0")
    ctrl_count = sum(1 for r in rows if r["type"] == "1")
    data_count = sum(1 for r in rows if r["type"] == "2")
    beacon_count_total = sum(1 for r in rows if r["subtype"] == BEACON_SUBTYPE)
    mean_frame_len = byte_count / packet_count if packet_count else 0.0
    bssids_heard = {r["bssid"].lower() for r in rows if r["bssid"]}

    # --- per-BSSID breakdown ---
    by_bssid: dict[str, list[dict]] = {}
    for r in rows:
        if r["bssid"]:
            by_bssid.setdefault(r["bssid"].lower(), []).append(r)

    target_rows = by_bssid.get(target_mac, [])
    target_beacon_rssi = [r["rssi"] for r in target_rows if r["subtype"] == BEACON_SUBTYPE]
    target_beacon_times = sorted(r["time"] for r in target_rows if r["subtype"] == BEACON_SUBTYPE)
    target_beacon_gaps = [b - a for a, b in zip(target_beacon_times, target_beacon_times[1:])]

    rival_macs = [m for m in by_bssid if m != target_mac]
    rival_mean_rssi = [safe_mean([r["rssi"] for r in by_bssid[m]]) for m in rival_macs]
    rival_mean_rssi = [x for x in rival_mean_rssi if x is not None]
    rival_frame_counts = [len(by_bssid[m]) for m in rival_macs]

    features = {
        "feat_win_packet_count": packet_count,
        "feat_win_byte_count": byte_count,
        "feat_win_mean_frame_len": mean_frame_len,
        "feat_win_mgmt_count": mgmt_count,
        "feat_win_ctrl_count": ctrl_count,
        "feat_win_data_count": data_count,
        "feat_win_beacon_count": beacon_count_total,
        "feat_n_bssids_heard": len(bssids_heard),

        "feat_target_seen": int(bool(target_rows)),
        "feat_target_frame_count": len(target_rows),
        "feat_target_byte_count": sum(r["len"] for r in target_rows),
        "feat_target_beacon_count": len(target_beacon_rssi),
        "feat_target_rssi_mean": safe_mean(target_beacon_rssi),
        "feat_target_rssi_std": safe_std(target_beacon_rssi),
        "feat_target_rssi_last": target_beacon_rssi[-1] if target_beacon_rssi else None,
        "feat_target_beacon_gap_std": safe_std(target_beacon_gaps),

        "feat_rival_count": len(rival_macs),
        "feat_rival_best_rssi": max(rival_mean_rssi) if rival_mean_rssi else None,
        "feat_rival_mean_rssi": safe_mean(rival_mean_rssi),
        "feat_rival_mean_frame_count": safe_mean(rival_frame_counts),

        "label_throughput_mbps": meta["candidate"]["throughput_mbps"],

        "gt_run_id": meta["run_id"],
        "gt_rng_seed": meta["rng_seed"],
        "gt_n_aps": meta["params"]["n_aps"],
        "gt_n_stas": meta["params"]["n_stas"],
        "gt_target_ap": target_ap_index,
        "gt_ap_spacing": meta["params"]["ap_spacing"],
        "gt_candidate_distance": meta["params"]["candidate_distance"],
        "gt_candidate_angle_deg": meta["params"]["candidate_angle_deg"],
        "gt_jitter_std": meta["params"]["jitter_std"],
        "gt_n_channels": meta["params"]["n_channels"],
        "gt_packet_size": meta["params"]["packet_size"],
        "gt_interval_ms": meta["params"]["interval_ms"],
        "gt_candidate_start_time": candidate_start,
        "gt_sim_stop_time": meta["params"]["sim_stop_time"],
        "gt_target_ap_background_mbps": target_ap["background_mbps"],
        "gt_target_ap_sta_count": target_ap["sta_count"],
        "gt_target_ap_channel": target_ap["channel"],
    }
    return features


def process_run(run_dir: Path) -> dict | None:
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        return None
    pcaps = list(run_dir.glob("candidate-trace*.pcap"))
    if not pcaps:
        return None

    try:
        meta = json.loads(meta_path.read_text())
        return extract_features(pcaps[0], meta)
    except Exception as e:  # noqa: BLE001 - want to keep going across a whole sweep
        print(f"WARN: skipping {run_dir}: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", type=Path, help="directory containing run_* subdirectories")
    parser.add_argument("--out", type=Path, default=Path("data/dataset.csv"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    run_dirs = sorted(p for p in args.runs_dir.iterdir() if p.is_dir())
    print(f"Found {len(run_dirs)} run directories under {args.runs_dir}")

    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(process_run, run_dirs), 1):
            if result is not None:
                rows.append(result)
            if i % 20 == 0 or i == len(run_dirs):
                print(f"[{i}/{len(run_dirs)}] extracted {len(rows)} rows so far")

    if not rows:
        sys.exit("No rows extracted - nothing to write.")

    import pandas as pd  # local import so --help doesn't need pandas installed

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows x {len(df.columns)} cols to {args.out}")


if __name__ == "__main__":
    main()
