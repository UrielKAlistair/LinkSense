#!/usr/bin/env python3
"""Build the raw-frame corpus: every decoded frame as its own token.

TODO: chanbusy.csv never reaches this corpus. The binned corpus gives every step
its channel's busy fraction, and the aggregate table gives each channel's busy
fraction over its dwells, but no frame token carries carrier-sense busy time, so
the frame model trains on no busy information at all. A gap between the frame
and binned models can come from this missing input rather than from the level
of aggregation.

The binned corpora cut the scan into fixed slices and hand the model summary
statistics over each one. That is a hand-built representation and a lossy one:
a bin that saw no frame has no signal reading, and whatever is written in its
place propagates into every aggregate taken afterwards.

This corpus drops the aggregation step. A scan's observation is the list of
frames the sweeping radio decoded (every recorded frame with --all-channels), in
time order, each carrying what a real capture carries: transmission start time,
signal, duration, length, rate, type, retry and beacon flags. No binning and no
summary statistics over the frames: one that was not received is simply not a
token.

Identity is encoded RELATIONALLY, never absolutely. Given raw BSSIDs a model
would learn the MAC ordering the simulator happens to assign, so instead each
option carries a per-frame relation code: was this frame on the option's
channel, from its BSS, sent by the AP itself. The same frame therefore looks
different to different options, which is what lets one shared trace answer a
per-option question.

Run:
  python scripts/dataset/build_frame_corpus.py data/runs --out data/frames.npz
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.models.frames import (FRAME_SCHEMA_VERSION, REL_FROM_AP,  # noqa: E402
                                   REL_ON_CHANNEL, REL_SAME_BSS)
from scripts.dataset.projection import (TYPE_CTRL, TYPE_DATA, TYPE_MGMT,  # noqa: E402
                                        describe, project)
from scripts.dataset.scans import Option, find_options, find_scans  # noqa: E402

FRAME_FEATURES = (
    "time_fraction",       # when in the window the frame's transmission started
    "rssi_dbm",
    "duration_log1p",
    "length_log1p",
    "rate_log1p",
    "is_beacon",
    "is_retry",
    "is_mgmt",
    "is_ctrl",
    "is_data",
    # Time since the previous frame's transmission started, as a fraction of
    # 100 ms, clipped there. Frames of one aggregate share a start, so every
    # one after the first gets 0.0, as does the first frame of a scan - read it
    # with time_fraction rather than alone.
    "gap_since_previous",
)

# The relation bits live in scripts/models/frames.py, next to the embedding that
# consumes them, and are imported above.


def frame_rows(rows: list[dict], window: float) -> np.ndarray:
    """(n_frames, F) in time order. Empty input gives a (0, F) array."""
    if not rows:
        return np.zeros((0, len(FRAME_FEATURES)), dtype=np.float32)
    ordered = sorted(rows, key=lambda r: r["tx_start"])
    out = np.zeros((len(ordered), len(FRAME_FEATURES)), dtype=np.float32)
    previous = ordered[0]["tx_start"]
    for i, r in enumerate(ordered):
        out[i] = (
            r["tx_start"] / window if window else 0.0,
            r["rssi"],
            math.log1p(r["dur"]),
            math.log1p(r["len"]),
            math.log1p(r["rate"]),
            float(r["beacon"]),
            float(r["retry"]),
            float(r["cat"] == TYPE_MGMT),
            float(r["cat"] == TYPE_CTRL),
            float(r["cat"] == TYPE_DATA),
            min(1.0, (r["tx_start"] - previous) / 0.1),
        )
        previous = r["tx_start"]
    return out


def relation_codes(rows: list[dict], options: list[Option]) -> np.ndarray:
    """(n_options, n_frames) uint8 relation of each frame to each option."""
    ordered = sorted(rows, key=lambda r: r["tx_start"])
    codes = np.zeros((len(options), len(ordered)), dtype=np.uint8)
    channel_of = np.array([r["channel"] for r in ordered])
    bssid_of = [r["bssid"] for r in ordered]
    ta_of = [r["ta"] for r in ordered]
    for o, option in enumerate(options):
        same_bss = np.fromiter((b == option.mac for b in bssid_of), bool, len(ordered))
        from_ap = np.fromiter((a == option.mac for a in ta_of), bool, len(ordered))
        codes[o] = ((channel_of == option.channel) * REL_ON_CHANNEL
                    + same_bss * REL_SAME_BSS + from_ap * REL_FROM_AP)
    return codes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None,
                        help="default data/frames.npz, or data/frames_all_channels.npz "
                             "with --all-channels")
    parser.add_argument("--max-frames", type=int, default=16384,
                        help="cap per scan. A scan over the cap is thinned uniformly "
                             "across the window rather than truncated, so the span it "
                             "covers does not depend on how busy the medium was")
    parser.add_argument("--all-channels", action="store_true",
                        help="keep every channel for the whole window instead of "
                             "projecting onto one radio")
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error("--max-frames must be at least 1")
    out_path = args.out or Path("data/frames_all_channels.npz" if args.all_channels
                                else "data/frames.npz")
    print(f"projection: {describe(args.all_channels)}")

    scans = find_scans(args.runs_dir)
    built = []
    thinned = 0
    for number, scan in enumerate(scans, 1):
        projected = project(scan.recording, scan.scan_id, args.all_channels)
        options = find_options(scan, projected)
        if options is not None:
            ordered = sorted(projected.frames, key=lambda r: r["tx_start"])
            if len(ordered) > args.max_frames:
                # Thin uniformly and keep the order, so the retained span still
                # covers the whole window rather than just its busy start.
                keep = np.linspace(0, len(ordered) - 1, args.max_frames).round().astype(int)
                ordered = [ordered[i] for i in keep]
                thinned += 1
            params = options[0].metadata["params"]
            built.append({
                "scan_id": scan.scan_id, "topology_id": scan.topology_id,
                "n_aps": int(params["n_aps"]),
                "n_hotspots": int(params["n_hotspots"]),
                "window_s": projected.window,
                "frames": frame_rows(ordered, projected.window),
                "relations": relation_codes(ordered, options),
                "option_indices": np.array([option.index for option in options]),
                "labels": np.array([option.throughput_mbps for option in options],
                                   dtype=np.float32),
            })
        if number % 100 == 0 or number == len(scans):
            print(f"[{number}/{len(scans)}] scans")

    max_options = max(len(i["option_indices"]) for i in built)
    max_frames = max(i["frames"].shape[0] for i in built)
    n_scans = len(built)
    counts = np.array([i["frames"].shape[0] for i in built])
    print(f"frames per scan: min={counts.min()} median={int(np.median(counts))} "
          f"max={counts.max()}; {thinned} scans thinned to {args.max_frames}")

    frames = np.zeros((n_scans, max_frames, len(FRAME_FEATURES)), np.float32)
    relations = np.zeros((n_scans, max_options, max_frames), np.uint8)
    frame_mask = np.zeros((n_scans, max_frames), bool)
    labels = np.zeros((n_scans, max_options), np.float32)
    option_indices = np.full((n_scans, max_options), -1, np.int16)
    option_mask = np.zeros((n_scans, max_options), bool)

    for i, item in enumerate(built):
        n_f = item["frames"].shape[0]
        n_o = len(item["option_indices"])
        frames[i, :n_f] = item["frames"]
        relations[i, :n_o, :n_f] = item["relations"]
        frame_mask[i, :n_f] = True
        labels[i, :n_o] = item["labels"]
        option_indices[i, :n_o] = item["option_indices"]
        option_mask[i, :n_o] = True

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, schema_version=np.array(FRAME_SCHEMA_VERSION, dtype=np.int16),
        frames=frames, relations=relations, frame_mask=frame_mask,
        labels=labels, option_indices=option_indices,
        option_mask=option_mask,
        scan_ids=np.array([i["scan_id"] for i in built]),
        topology_ids=np.array([i["topology_id"] for i in built]),
        configured_n_aps=np.array([i["n_aps"] for i in built], dtype=np.int16),
        n_hotspots=np.array([i["n_hotspots"] for i in built], dtype=np.int16),
        window_s=np.array([i["window_s"] for i in built], dtype=np.float32),
        frame_features=np.array(FRAME_FEATURES),
        scan_description=np.array(describe(args.all_channels)))
    print(f"wrote {n_scans} scans x {max_frames} frames x {len(FRAME_FEATURES)} "
          f"features to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
