#!/usr/bin/env python3
"""Build the raw-frame corpus: every decoded frame as its own token.

The binned corpora cut the scan into fixed slices and hand the model summary
statistics over each one. That is a hand-built representation and a lossy one:
a bin that saw no frame has no signal reading, and whatever is written in its
place propagates into every aggregate taken afterwards.

This corpus drops the aggregation step. A scan's observation is the list of
frames the sweeping radio decoded, in time order, each carrying what a real
capture carries: arrival time, signal, noise, duration, length, rate, type,
retry and beacon flags. No binning, no summary statistics, no imputation - a
frame that was not received is simply not a token.

Identity is encoded RELATIONALLY, never absolutely. Given raw BSSIDs a model
would learn the MAC ordering the simulator happens to assign, so instead each
option carries a per-frame relation code: was this frame on the option's
channel, from its BSS, sent by the AP itself. The same frame therefore looks
different to different options, which is what lets one shared trace answer a
per-option question.

Run:
  python scripts/dataset/build_frame_corpus.py data/runs data/dataset.csv \
      --out data/frames.npz
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.data import feature_columns, impute_features, load_dataset  # noqa: E402
from models.frames import REL_FROM_AP, REL_ON_CHANNEL, REL_SAME_BSS  # noqa: E402
from scripts.dataset.build_dataset import (SCAN_CHANNELS, TYPE_CTRL,  # noqa: E402
                                   TYPE_DATA, TYPE_MGMT, ScanConfig, _rng,
                                   apply_rssi_realism, read_observation,
                                   reference_dir, single_radio_sweep)

FRAME_FEATURES = (
    "time_fraction",       # when in the window the frame ended
    "rssi_dbm",
    "snr_db",
    "duration_log1p",
    "length_log1p",
    "rate_log1p",
    "is_beacon",
    "is_retry",
    "is_mgmt",
    "is_ctrl",
    "is_data",
    "gap_since_previous",  # seconds since the previous decoded frame
)

# The relation bits live in models/frames.py, next to the embedding that
# consumes them, and are imported above.


def frame_rows(rows: list[dict], window: float) -> np.ndarray:
    """(n_frames, F) in time order. Empty input gives a (0, F) array."""
    if not rows:
        return np.zeros((0, len(FRAME_FEATURES)), dtype=np.float32)
    ordered = sorted(rows, key=lambda r: r["t"])
    out = np.zeros((len(ordered), len(FRAME_FEATURES)), dtype=np.float32)
    previous = ordered[0]["t"]
    for i, r in enumerate(ordered):
        out[i] = (
            r["t"] / window if window else 0.0,
            r["rssi"],
            r["rssi"] - r["noise"],
            math.log1p(r["dur"]),
            math.log1p(r["len"]),
            math.log1p(r["rate"]),
            float(r["beacon"]),
            float(r["retry"]),
            float(r["cat"] == TYPE_MGMT),
            float(r["cat"] == TYPE_CTRL),
            float(r["cat"] == TYPE_DATA),
            min(1.0, (r["t"] - previous) / 0.1),
        )
        previous = r["t"]
    return out


def relation_codes(rows: list[dict], option_meta: list[dict]) -> np.ndarray:
    """(n_options, n_frames) uint8 relation of each frame to each option."""
    ordered = sorted(rows, key=lambda r: r["t"])
    codes = np.zeros((len(option_meta), len(ordered)), dtype=np.uint8)
    channel_of = np.array([(r["freq"] - 5000) // 5 for r in ordered])
    bssid_of = [r["bssid"] for r in ordered]
    ta_of = [r["ta"] for r in ordered]
    for o, ap in enumerate(option_meta):
        same_bss = np.fromiter((b == ap["mac"] for b in bssid_of), bool, len(ordered))
        from_ap = np.fromiter((a == ap["mac"] for a in ta_of), bool, len(ordered))
        codes[o] = ((channel_of == ap["channel"]) * REL_ON_CHANNEL
                    + same_bss * REL_SAME_BSS + from_ap * REL_FROM_AP)
    return codes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_dir", type=Path, nargs="+")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/frames.npz"))
    parser.add_argument("--max-frames", type=int, default=16384,
                        help="cap per scan. A scan over the cap is thinned uniformly "
                             "across the window rather than truncated, so the span it "
                             "covers does not depend on how busy the medium was")
    args = parser.parse_args()

    frame = impute_features(load_dataset(args.dataset))
    if frame["meta_scan_dwell_ms"].isna().any():
        raise ValueError("frame corpus requires a --single-radio-sweep dataset")
    value = {c: frame[c].dropna().unique()[0] for c in (
        "meta_scan_dwell_ms", "meta_scan_retune_ms", "meta_scan_passes",
        "meta_scan_order", "meta_scan_align", "meta_scan_seed",
        "meta_rssi_noise_db", "meta_rssi_noise_model", "meta_rssi_bias_db",
        "meta_rssi_quant_db")}
    cfg = ScanConfig(
        sweep=True, mode="passive",
        dwell_ms=float(value["meta_scan_dwell_ms"]),
        retune_ms=float(value["meta_scan_retune_ms"]),
        passes=int(value["meta_scan_passes"]),
        order=str(value["meta_scan_order"]), align=str(value["meta_scan_align"]),
        rssi_noise_db=float(value["meta_rssi_noise_db"]),
        rssi_noise_model=str(value["meta_rssi_noise_model"]),
        rssi_bias_db=float(value["meta_rssi_bias_db"]),
        rssi_quant_db=float(value["meta_rssi_quant_db"]),
        seed=int(value["meta_scan_seed"]))
    print(f"observation model: {cfg.describe()}")

    static_features = feature_columns(frame)
    built = []
    total = frame["scan_id"].nunique()
    truncated = 0
    for number, (scan_id, rows_of_scan) in enumerate(frame.groupby("scan_id", sort=True), 1):
        rows_of_scan = rows_of_scan.sort_values("ap_index")
        topology_id = str(rows_of_scan["topology_id"].iloc[0])
        scan_key = scan_id.split("__", 1)[1]
        ref = reference_dir(args.runs_dir, topology_id, scan_key)
        meta = json.loads((ref / "metadata.json").read_text())
        window = float(meta["params"]["feature_window_end"])

        ap_by_index = {int(ap["index"]): {"index": int(ap["index"]),
                                          "mac": ap["mac"].lower(),
                                          "channel": int(ap["channel"])}
                       for ap in meta["aps"]}
        freq_of_chan = {ch: 5000 + 5 * ch for ch in SCAN_CHANNELS}

        rows = read_observation(ref / "observation.csv")
        apply_rssi_realism(rows, cfg, _rng(cfg, scan_id, "rssi"))
        rows, _, _, _, _ = single_radio_sweep(
            rows, freq_of_chan, window, cfg, _rng(cfg, scan_id, "sweep"))

        option_indices = rows_of_scan["ap_index"].astype(int).to_numpy()
        option_meta = [ap_by_index[i] for i in option_indices]

        ordered = sorted(rows, key=lambda r: r["t"])
        if len(ordered) > args.max_frames:
            # Thin uniformly and keep the order, so the retained span still
            # covers the whole window rather than just its busy start.
            keep = np.linspace(0, len(ordered) - 1, args.max_frames).round().astype(int)
            ordered = [ordered[i] for i in np.unique(keep)]
            truncated += 1
        built.append({
            "scan_id": scan_id, "topology_id": topology_id,
            "n_aps": int(rows_of_scan["gt_n_aps"].iloc[0]),
            "n_hotspots": int(rows_of_scan["gt_n_hotspots"].iloc[0]),
            "candidate_stratum": str(rows_of_scan["gt_candidate_stratum"].iloc[0]),
            "frames": frame_rows(ordered, window),
            "relations": relation_codes(ordered, option_meta),
            "static": rows_of_scan[static_features].to_numpy(dtype=np.float32),
            "option_indices": option_indices,
            "labels": rows_of_scan["label_throughput_mbps"].to_numpy(dtype=np.float32),
        })
        if number % 100 == 0 or number == total:
            print(f"[{number}/{total}] scans, {len(ordered)} frames in the last one")

    max_options = max(len(i["option_indices"]) for i in built)
    max_frames = max(i["frames"].shape[0] for i in built)
    n_scans = len(built)
    counts = np.array([i["frames"].shape[0] for i in built])
    print(f"frames per scan: min={counts.min()} median={int(np.median(counts))} "
          f"max={counts.max()}; {truncated} scans truncated at {args.max_frames}")

    frames = np.zeros((n_scans, max_frames, len(FRAME_FEATURES)), np.float32)
    relations = np.zeros((n_scans, max_options, max_frames), np.uint8)
    frame_mask = np.zeros((n_scans, max_frames), bool)
    static = np.zeros((n_scans, max_options, len(static_features)), np.float32)
    labels = np.zeros((n_scans, max_options), np.float32)
    option_indices = np.full((n_scans, max_options), -1, np.int16)
    option_mask = np.zeros((n_scans, max_options), bool)

    for i, item in enumerate(built):
        n_f = item["frames"].shape[0]
        n_o = len(item["option_indices"])
        frames[i, :n_f] = item["frames"]
        relations[i, :n_o, :n_f] = item["relations"]
        frame_mask[i, :n_f] = True
        static[i, :n_o] = item["static"]
        labels[i, :n_o] = item["labels"]
        option_indices[i, :n_o] = item["option_indices"]
        option_mask[i, :n_o] = True

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, schema_version=np.array(1, dtype=np.int16),
        frames=frames, relations=relations, frame_mask=frame_mask,
        static=static, labels=labels, option_indices=option_indices,
        option_mask=option_mask,
        scan_ids=np.array([i["scan_id"] for i in built]),
        topology_ids=np.array([i["topology_id"] for i in built]),
        configured_n_aps=np.array([i["n_aps"] for i in built], dtype=np.int16),
        n_hotspots=np.array([i["n_hotspots"] for i in built], dtype=np.int16),
        candidate_strata=np.array([i["candidate_stratum"] for i in built]),
        frame_features=np.array(FRAME_FEATURES),
        static_features=np.array(static_features),
        scan_description=np.array(cfg.describe()))
    print(f"wrote {n_scans} scans x {max_frames} frames x {len(FRAME_FEATURES)} "
          f"features to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
