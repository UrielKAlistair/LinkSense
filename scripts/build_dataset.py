#!/usr/bin/env python3
"""Turn matched-set ns-3 runs into a choice-set training table.

One row per (scenario, AP option). Rows sharing a group_id describe the
same physical situation and the same pre-association observation; they
differ only in which AP the candidate joined, and therefore in the label.

Everything under feat_ is derived solely from frames the candidate's radio
captured in [0, candidate_start_time) - before it transmits anything or
joins anything. Three families:

  feat_env_*  channel-wide conditions (identical across a group's rows)
  feat_ap_*   what was observed about THIS option's BSS
  feat_rel_*  this option relative to the alternatives (RSSI rank, margin
              over the best other AP, share of observed airtime). These
              are what let a model compare options rather than score them
              in isolation, and they are the features a ranking objective
              actually keys on.

Airtime comes from the simulator's own per-frame TX duration and is a
legitimate observable:
real chipsets measure medium occupancy, and 802.11k/QBSS BSS-Load reports
expose per-BSS channel utilisation to clients before association.

gt_* columns are simulator ground truth (true distances, true offered
load, seeds). They exist for validating the feat_* proxies and for
stratifying analysis, and must never be model inputs - a real station
cannot observe them. models/data.py enforces this.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import csv
import sys
from collections import defaultdict
from pathlib import Path

TYPE_MGMT, TYPE_CTRL, TYPE_DATA = 0, 1, 2


def read_observation(obs_csv: Path) -> list[dict]:
    """Read the observation the simulator recorded from its scanner radios.

    The simulator already restricts this to the guarded pre-association
    window and reports exact signal/noise and per-frame airtime, so there is
    nothing to filter or reconstruct here.
    """
    rows = []
    with obs_csv.open(newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "t": float(r["t"]),
                "freq": int(r["freq_mhz"]),
                "bssid": r["bssid"].lower() or None,
                "ta": r["ta"].lower() or None,
                "cat": int(r["cat"]),
                "beacon": r["is_beacon"] == "1",
                "retry": r["retry"] == "1",
                "len": int(r["len"]),
                "rssi": float(r["signal_dbm"]),
                "noise": float(r["noise_dbm"]),
                "dur": float(r["duration_us"]),
                "rate": float(r["rate_mbps"]),
            })
    return rows


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def _std(xs):
    xs = [x for x in xs if x is not None]
    return statistics.pstdev(xs) if len(xs) > 1 else (0.0 if xs else None)


def channel_features(rows: list[dict], window: float) -> tuple[dict, dict[str, dict]]:
    """Return (channel-level features, per-BSSID features) for one channel."""
    total_airtime_us = sum(r["dur"] for r in rows)
    data_rows = [r for r in rows if r["cat"] == TYPE_DATA]

    # Channel-level conditions. With APs spread across channels these differ
    # per option, and they are the load signal that makes AP choice more than
    # a signal-strength comparison.
    chan = {
        "feat_chan_busy_frac": total_airtime_us / (window * 1e6) if window else 0.0,
        "feat_chan_frames_per_s": len(rows) / window if window else 0.0,
        "feat_chan_bytes_per_s": sum(r["len"] for r in rows) / window if window else 0.0,
        "feat_chan_data_frac": len(data_rows) / len(rows) if rows else 0.0,
        "feat_chan_mgmt_frac": sum(1 for r in rows if r["cat"] == TYPE_MGMT) / len(rows) if rows else 0.0,
        "feat_chan_ctrl_frac": sum(1 for r in rows if r["cat"] == TYPE_CTRL) / len(rows) if rows else 0.0,
        "feat_chan_retry_frac": sum(1 for r in rows if r["retry"]) / len(rows) if rows else 0.0,
        "feat_chan_mean_data_rate": _mean([r["rate"] for r in data_rows]) or 0.0,
        "feat_chan_n_bssids": len({r["bssid"] for r in rows if r["bssid"]}),
        "feat_chan_n_tas": len({r["ta"] for r in rows if r["ta"]}),
        "feat_chan_mean_rssi": _mean([r["rssi"] for r in rows]) or -100.0,
        "_airtime_us": total_airtime_us,
    }

    by_bssid: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["bssid"]:
            by_bssid[r["bssid"]].append(r)

    per_ap: dict[str, dict] = {}
    for bssid, frames in by_bssid.items():
        beacons = [f for f in frames if f["beacon"]]
        beacon_rssi = [f["rssi"] for f in beacons]
        btimes = sorted(f["t"] for f in beacons)
        gaps = [b - a for a, b in zip(btimes, btimes[1:])]
        airtime = sum(f["dur"] for f in frames)
        # transmitters in this BSS other than the AP itself: a client-count
        # estimate a scanning radio really can form
        clients = {f["ta"] for f in frames if f["ta"] and f["ta"] != bssid}
        dframes = [f for f in frames if f["cat"] == TYPE_DATA]

        per_ap[bssid] = {
            "feat_ap_rssi_mean": _mean(beacon_rssi),
            "feat_ap_rssi_std": _std(beacon_rssi),
            "feat_ap_rssi_max": max(beacon_rssi) if beacon_rssi else None,
            "feat_ap_rssi_min": min(beacon_rssi) if beacon_rssi else None,
            "feat_ap_rssi_last": beacon_rssi[-1] if beacon_rssi else None,
            "feat_ap_beacons": len(beacons),
            "feat_ap_beacon_gap_mean": _mean(gaps),
            "feat_ap_beacon_gap_std": _std(gaps),
            "feat_ap_frames": len(frames),
            "feat_ap_frames_per_s": len(frames) / window if window else 0.0,
            "feat_ap_bytes_per_s": sum(f["len"] for f in frames) / window if window else 0.0,
            "feat_ap_airtime_frac": airtime / (window * 1e6) if window else 0.0,
            "feat_ap_n_clients": len(clients),
            "feat_ap_data_frames": len(dframes),
            "feat_ap_mean_data_rate": _mean([f["rate"] for f in dframes]) or 0.0,
            "feat_ap_retry_frac": sum(1 for f in frames if f["retry"]) / len(frames) if frames else 0.0,
            "_airtime_us": airtime,
        }
    return chan, per_ap


def build_group_rows(group_id: str, variants: list[tuple[int, Path, dict]]) -> list[dict]:
    """variants: (target_ap, run_dir, metadata) for every AP option in a group.

    The observation is identical across a group's variants (the simulator
    parks the association radio so the choice cannot influence what was
    observed; verified byte-for-byte in validate_sim.py V7), so it is
    recorded for one variant only and shared here. Only the labels come from
    the individual variants.
    """
    variants = sorted(variants)
    ref_dir = next((d for _, d, _ in variants if (d / "observation.csv").exists()), None)
    if ref_dir is None:
        raise FileNotFoundError("no variant in this group recorded an observation")
    ref_meta = next(m for _, d, m in variants if d == ref_dir)
    window = ref_meta["params"]["feature_window_end"]

    ap_meta = {ap["index"]: ap for ap in ref_meta["aps"]}
    mac_of = {ap["index"]: ap["mac"].lower() for ap in ref_meta["aps"]}
    chan_of = {ap["index"]: int(ap["channel"]) for ap in ref_meta["aps"]}

    # Split the observation by the frequency it was heard on: each scanner
    # radio watches one channel, so frequency identifies the channel whose
    # conditions an option would actually experience.
    rows = read_observation(ref_dir / "observation.csv")
    freq_of_chan = {ch: 5000 + 5 * ch for ch in set(chan_of.values())}
    by_freq: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_freq[r["freq"]].append(r)

    chan_feats: dict[int, dict] = {}
    per_ap: dict[str, dict] = {}
    for ch in sorted(set(chan_of.values())):
        cf, ap_f = channel_features(by_freq.get(freq_of_chan[ch], []), window)
        chan_feats[ch] = cf
        per_ap.update(ap_f)

    # Comparisons are made across every AP the client can hear, on any
    # channel - which is what a real scan yields and what the choice is
    # actually between.
    heard = {idx: per_ap[mac]["feat_ap_rssi_mean"]
             for idx, mac in mac_of.items()
             if mac in per_ap and per_ap[mac]["feat_ap_rssi_mean"] is not None}
    total_airtime = sum(v["_airtime_us"] for v in per_ap.values()) or 1.0

    rows = []
    for target_ap, _dir, meta in variants:
        mac = mac_of[target_ap]
        chan = dict(chan_feats[chan_of[target_ap]])
        chan.pop("_airtime_us", None)
        apf = dict(per_ap.get(mac, {}))
        apf.pop("_airtime_us", None)
        seen = mac in per_ap
        if not seen:  # never heard this AP at all
            apf = {k: None for k in (
                "feat_ap_rssi_mean", "feat_ap_rssi_std", "feat_ap_rssi_max",
                "feat_ap_rssi_min", "feat_ap_rssi_last", "feat_ap_beacon_gap_mean",
                "feat_ap_beacon_gap_std")}
            apf.update({k: 0.0 for k in (
                "feat_ap_beacons", "feat_ap_frames", "feat_ap_frames_per_s",
                "feat_ap_bytes_per_s", "feat_ap_airtime_frac", "feat_ap_n_clients",
                "feat_ap_data_frames", "feat_ap_mean_data_rate", "feat_ap_retry_frac")})

        mine = heard.get(target_ap)
        others = [v for k, v in heard.items() if k != target_ap]
        best_other = max(others) if others else None

        rel = {
            "feat_rel_seen": int(seen),
            "feat_rel_n_options": len(mac_of),
            "feat_rel_n_heard": len(heard),
            "feat_rel_rssi_rank": (
                sorted(heard.values(), reverse=True).index(mine) if mine is not None else -1),
            "feat_rel_is_strongest": int(mine is not None and mine == max(heard.values())) if heard else 0,
            "feat_rel_rssi_margin_best_other": (
                mine - best_other if (mine is not None and best_other is not None) else None),
            "feat_rel_rssi_minus_mean": (
                mine - statistics.fmean(heard.values()) if mine is not None and heard else None),
            "feat_rel_airtime_share": (
                per_ap[mac]["_airtime_us"] / total_airtime if seen else 0.0),
            "feat_rel_clients_share": None,
        }
        tot_clients = sum(per_ap[m]["feat_ap_n_clients"] for m in per_ap)
        rel["feat_rel_clients_share"] = (
            per_ap[mac]["feat_ap_n_clients"] / tot_clients if seen and tot_clients else 0.0)

        c = meta["candidate"]
        gt_ap = ap_meta[target_ap]
        row = {
            "group_id": group_id,
            "ap_index": target_ap,
            "run_id": meta["run_id"],
            "feat_env_window_s": window,
            **chan, **apf, **rel,
            "label_throughput_mbps": c["throughput_mbps"],
            "label_associated": int(c["associated"]),
            "gt_rng_seed": meta["rng_seed"],
            "gt_n_aps": meta["params"]["n_aps"],
            "gt_n_stas": meta["params"]["n_stas"],
            "gt_ap_spacing": meta["params"]["ap_spacing"],
            "gt_bg_per_sta_mbps": meta["params"]["bg_per_sta_mbps"],
            "gt_bg_total_offered_mbps": meta["params"]["bg_total_offered_mbps"],
            "gt_packet_size": meta["params"]["packet_size"],
            "gt_jitter_std": meta["params"]["jitter_std"],
            "gt_candidate_x": meta["candidate_position"]["x"],
            "gt_candidate_y": meta["candidate_position"]["y"],
            "gt_true_distance": gt_ap["candidate_distance"],
            "gt_ap_sta_count": gt_ap["sta_count"],
            "gt_ap_offered_mbps": gt_ap["offered_mbps"],
            "gt_ap_background_mbps": gt_ap["background_mbps"],
            "gt_assoc_delay": c["assoc_delay"],
            "gt_observed_seconds": c["observed_seconds"],
            "gt_n_channels": meta["params"]["n_channels"],
            "gt_ap_channel": chan_of[target_ap],
        }
        rows.append(row)
    return rows


def process_group(item) -> list[dict]:
    group_id, run_dirs = item
    variants = []
    for d in run_dirs:
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        variants.append((meta["candidate"]["target_ap"], d, meta))
    if not variants:
        return []
    # a partially-failed group would give a ranking task a truncated option
    # set, which silently changes what "best AP" means - drop it instead
    expected = variants[0][2]["params"]["n_aps"]
    if len(variants) != expected:
        print(f"WARN: group {group_id} has {len(variants)}/{expected} variants; skipping",
              file=sys.stderr)
        return []
    try:
        return build_group_rows(group_id, variants)
    except Exception as e:  # noqa: BLE001
        print(f"WARN: group {group_id} failed: {e}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/dataset.csv"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    groups: dict[str, list[Path]] = defaultdict(list)
    for d in sorted(p for p in args.runs_dir.iterdir() if p.is_dir()):
        groups[d.name.split("__")[0]].append(d)
    print(f"{len(groups)} groups under {args.runs_dir}")

    rows: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, res in enumerate(pool.map(process_group, list(groups.items())), 1):
            rows.extend(res)
            if i % 50 == 0 or i == len(groups):
                print(f"[{i}/{len(groups)}] {len(rows)} rows")

    if not rows:
        sys.exit("no rows extracted")

    import pandas as pd

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows x {len(df.columns)} cols "
          f"({df.group_id.nunique()} groups) to {args.out}")


if __name__ == "__main__":
    main()
