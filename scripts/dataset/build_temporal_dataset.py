#!/usr/bin/env python3
"""Build the ordered, binned view of a scan for temporal models.

build_dataset.py collapses each scan to 36 scalars per option. This keeps the
time axis: the same single-radio sweep, cut into short bins, with every option
given the shared channel context plus the activity attributable to its own BSS
in each bin.

The output covers the same scans, options and labels as the tabular table, so
the two differ only in what a model is handed. That is the comparison the
corpus exists to support.

A real client tunes one channel at a time, so an option's own BSS is visible
only during the dwells on its channel. build_continuous_temporal.py lifts that
restriction, to separate "the trajectory carries nothing" from "the dwell was
too short to see it".

Run:
  python scripts/dataset/build_temporal_dataset.py data/runs data/dataset.csv \
      --out data/temporal.npz
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
from scripts.dataset.build_dataset import (SCAN_CHANNELS, TYPE_DATA,  # noqa: E402
                                   ScanConfig, _rng, apply_rssi_realism,
                                   read_chanbusy, read_observation,
                                   reference_dir, single_radio_sweep)


# One summary block, in the order _frame_summary() emits it. The last two are
# not measurements: `observed` says whether the bin carried any frame from the
# subject, and `age` how long ago the level readings were actually taken.
SUMMARY_NAMES = ("frames_log1p", "bytes_log1p", "airtime_fraction",
                 "rssi_mean", "rssi_max", "beacons_log1p", "data_fraction",
                 "retry_fraction", "rate_log1p", "transmitters_log1p",
                 "observed", "age")
SUMMARY_WIDTH = len(SUMMARY_NAMES)
# offsets within a block
LEVEL_OFFSETS = (3, 4)      # rssi_mean, rssi_max: absolute dBm
OBSERVED_OFFSET = 10
AGE_OFFSET = 11
NOISE_FLOOR_DBM = -95.0

PREFIX_FEATURES = ("time_fraction", "dwell_fraction", "is_option_channel",
                   "option_share_on_channel", "cca_busy_fraction")

TEMPORAL_FEATURES = (
    PREFIX_FEATURES
    + tuple(f"channel_{n}" for n in SUMMARY_NAMES)
    + tuple(f"option_{n}" for n in SUMMARY_NAMES)
)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _frame_summary(frames: list[dict], listen_s: float) -> list[float]:
    """One summary block for one time bin; levels are NaN when nothing was heard.

    An empty bin has no signal reading, so the level columns are NaN rather than
    a sentinel. A sentinel would sit far below anything physical, dominate the
    standardiser, and turn the time-average of the column into a count of how
    many bins held a frame. _fill_levels() resolves the NaN once the whole
    series exists, and `observed`/`age` preserve what that fill would erase.
    """
    if not frames:
        nan = float("nan")
        return [0.0, 0.0, 0.0, nan, nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    count = len(frames)
    rssis = [r["rssi"] for r in frames]
    data = [r for r in frames if r["cat"] == TYPE_DATA]
    transmitters = {r["ta"] for r in frames if r["ta"]}
    return [
        math.log1p(count),
        math.log1p(sum(r["len"] for r in frames)),
        min(1.0, sum(r["dur"] for r in frames) / (listen_s * 1e6)) if listen_s else 0.0,
        float(np.mean(rssis)),
        max(rssis),
        math.log1p(sum(r["beacon"] for r in frames)),
        len(data) / count,
        sum(r["retry"] for r in frames) / count,
        math.log1p(float(np.mean([r["rate"] for r in data]))) if data else 0.0,
        math.log1p(len(transmitters)),
        1.0,   # observed
        0.0,   # age, filled in by _fill_levels
    ]


def _fill_levels(out: np.ndarray, block_starts: tuple[int, ...]) -> None:
    """Resolve NaN level readings per option and record how stale each one is.

    Forward fill is what a scan cache does: the last measurement stands until a
    newer one replaces it. The leading gap before an option's first reading is
    back-filled, so nothing outside the physical range enters the tensor, and
    `age` marks those bins as not live.
    """
    n_options, n_steps, _ = out.shape
    steps = np.arange(n_steps)
    for base in block_starts:
        first_level = base + LEVEL_OFFSETS[0]
        seen = ~np.isnan(out[:, :, first_level])
        for option in range(n_options):
            valid = np.flatnonzero(seen[option])
            if len(valid) == 0:
                for offset in LEVEL_OFFSETS:
                    out[option, :, base + offset] = NOISE_FLOOR_DBM
                out[option, :, base + AGE_OFFSET] = 1.0
                continue
            last = np.where(seen[option], steps, -1)
            np.maximum.accumulate(last, out=last)
            source = np.where(last >= 0, last, valid[0])
            for offset in LEVEL_OFFSETS:
                out[option, :, base + offset] = out[option, source, base + offset]
            age = np.where(last >= 0, steps - last, n_steps)
            out[option, :, base + AGE_OFFSET] = np.minimum(age / max(1, n_steps), 1.0)
        out[:, :, base + OBSERVED_OFFSET] = seen


def scan_sequence(rows: list[dict], busy: list[dict], slots: list[int], t0: float,
                   cfg: ScanConfig, option_meta: list[dict], bin_ms: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Return (N options, T bins, F features) and the valid-time mask."""
    bins_per_dwell = max(1, math.ceil(cfg.dwell / bin_ms))
    bin_s = cfg.dwell_s / bins_per_dwell
    n_steps = len(slots) * bins_per_dwell
    out = np.zeros((len(option_meta), n_steps, len(TEMPORAL_FEATURES)), dtype=np.float32)
    time_mask = np.ones(n_steps, dtype=bool)

    frames_by_step: list[list[dict]] = [[] for _ in range(n_steps)]
    for frame in rows:
        start = frame["t"] - frame["dur"] / 1e6
        # floor, not int: a frame starting before the window has a negative
        # offset, and int() truncates toward zero, landing it in bin 0 instead
        # of being rejected by the guard below.
        step = math.floor((start - t0) / bin_s) if bin_s else -1
        if 0 <= step < n_steps:
            frames_by_step[step].append(frame)

    busy_by_channel: dict[int, list[dict]] = {ch: [] for ch in set(slots)}
    for event in busy:
        if event["channel"] in busy_by_channel:
            busy_by_channel[event["channel"]].append(event)

    option_count_by_channel: dict[int, int] = {}
    for ap in option_meta:
        option_count_by_channel[ap["channel"]] = option_count_by_channel.get(ap["channel"], 0) + 1

    retune_s = cfg.retune_ms / 1000.0
    for step in range(n_steps):
        slot_index = step // bins_per_dwell
        within_slot = step % bins_per_dwell
        channel = slots[slot_index]
        start = t0 + step * bin_s
        end = start + bin_s
        slot_start = t0 + slot_index * cfg.dwell_s
        listen_start = max(start, slot_start + retune_s)
        listen_s = max(0.0, end - listen_start)

        frames = frames_by_step[step]
        channel_summary = _frame_summary(frames, listen_s)
        busy_s = sum(
            _overlap(event["start"], event["end"], listen_start, end) * event["busy_frac"]
            for event in busy_by_channel.get(channel, [])
        )
        cca = min(1.0, busy_s / listen_s) if listen_s else 0.0

        for option_index, ap in enumerate(option_meta):
            option_frames = [r for r in frames if r["bssid"] == ap["mac"]]
            prefix = [
                (step + 0.5) / max(1, n_steps),
                (within_slot + 0.5) / bins_per_dwell,
                float(channel == ap["channel"]),
                option_count_by_channel.get(channel, 0) / len(option_meta),
                cca,
            ]
            out[option_index, step] = prefix + channel_summary + \
                _frame_summary(option_frames, listen_s)

    _fill_levels(out, (len(PREFIX_FEATURES),
                       len(PREFIX_FEATURES) + SUMMARY_WIDTH))
    assert not np.isnan(out).any(), "unresolved NaN in temporal tensor"
    return out, time_mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", type=Path, nargs="+")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/temporal.npz"))
    parser.add_argument("--bin-ms", type=float, default=10.0,
                        help="nominal time-bin width within each dwell (default 10 ms)")
    args = parser.parse_args()

    if args.bin_ms <= 0:
        parser.error("--bin-ms must be positive")

    frame = impute_features(load_dataset(args.dataset))
    if set(frame["meta_scan_mode"].dropna().unique()) != {"passive"}:
        raise ValueError("temporal dataset currently requires a passive single-radio dataset")
    if frame["meta_scan_dwell_ms"].isna().any():
        raise ValueError("dataset was not built with --single-radio-sweep")

    provenance = {
        column: frame[column].dropna().unique()
        for column in (
            "meta_scan_dwell_ms", "meta_scan_retune_ms", "meta_scan_passes",
            "meta_scan_order", "meta_scan_align", "meta_scan_seed",
            "meta_rssi_noise_db", "meta_rssi_noise_model", "meta_rssi_bias_db",
            "meta_rssi_quant_db",
        )
    }
    nonconstant = [column for column, values in provenance.items() if len(values) != 1]
    if nonconstant:
        raise ValueError(f"scan provenance must be present and constant: {nonconstant}")
    value = {column: values[0] for column, values in provenance.items()}
    cfg = ScanConfig(
        sweep=True,
        mode="passive",
        dwell_ms=float(value["meta_scan_dwell_ms"]),
        retune_ms=float(value["meta_scan_retune_ms"]),
        passes=int(value["meta_scan_passes"]),
        order=str(value["meta_scan_order"]),
        align=str(value["meta_scan_align"]),
        rssi_noise_db=float(value["meta_rssi_noise_db"]),
        rssi_noise_model=str(value["meta_rssi_noise_model"]),
        rssi_bias_db=float(value["meta_rssi_bias_db"]),
        rssi_quant_db=float(value["meta_rssi_quant_db"]),
        seed=int(value["meta_scan_seed"]),
    )

    static_features = feature_columns(frame)
    built = []
    for number, (scan_id, rows_of_scan) in enumerate(frame.groupby("scan_id", sort=True), 1):
        rows_of_scan = rows_of_scan.sort_values("ap_index")
        topology_id = str(rows_of_scan["topology_id"].iloc[0])
        scan_key = scan_id.split("__", 1)[1]
        ref_dir = reference_dir(args.runs_dir, topology_id, scan_key)
        meta = json.loads((ref_dir / "metadata.json").read_text())

        ap_by_index = {int(ap["index"]): {
            "index": int(ap["index"]),
            "mac": ap["mac"].lower(),
            "channel": int(ap["channel"]),
        } for ap in meta["aps"]}
        freq_of_chan = {ch: 5000 + 5 * ch for ch in SCAN_CHANNELS}

        rows = read_observation(ref_dir / "observation.csv")
        apply_rssi_realism(rows, cfg, _rng(cfg, scan_id, "rssi"))
        rows, _, _, slots, t0 = single_radio_sweep(
            rows, freq_of_chan, meta["params"]["feature_window_end"], cfg,
            _rng(cfg, scan_id, "sweep"))
        assert slots is not None

        option_indices = rows_of_scan["ap_index"].astype(int).to_numpy()
        option_meta = [ap_by_index[index] for index in option_indices]
        sequence, time_mask = scan_sequence(
            rows, read_chanbusy(ref_dir / "chanbusy.csv"), slots, t0, cfg,
            option_meta, args.bin_ms)

        # The tabular builder admits an option only if a beacon decoded.
        # Rechecking here catches scan-seed or provenance drift between the two.
        beacon_macs = {r["bssid"] for r in rows if r["beacon"] and r["bssid"]}
        missing = [ap["index"] for ap in option_meta if ap["mac"] not in beacon_macs]
        if missing:
            raise ValueError(
                f"{scan_id}: tabular options missing from temporal scan: {missing}")

        built.append({
            "scan_id": scan_id,
            "topology_id": topology_id,
            "n_aps": int(rows_of_scan["gt_n_aps"].iloc[0]),
            "n_hotspots": int(rows_of_scan["gt_n_hotspots"].iloc[0]),
            "candidate_stratum": str(rows_of_scan["gt_candidate_stratum"].iloc[0]),
            "sequence": sequence,
            "time_mask": time_mask,
            "static": rows_of_scan[static_features].to_numpy(dtype=np.float32),
            "option_indices": option_indices,
            "labels": rows_of_scan["label_throughput_mbps"].to_numpy(dtype=np.float32),
        })
        if number % 25 == 0 or number == frame["scan_id"].nunique():
            print(f"[{number}/{frame['scan_id'].nunique()}] temporal scans")

    max_options = max(len(item["option_indices"]) for item in built)
    max_steps = max(item["sequence"].shape[1] for item in built)
    n_scans = len(built)
    n_temporal = len(TEMPORAL_FEATURES)
    n_static = len(static_features)

    temporal = np.zeros((n_scans, max_options, max_steps, n_temporal), dtype=np.float32)
    static = np.zeros((n_scans, max_options, n_static), dtype=np.float32)
    labels = np.zeros((n_scans, max_options), dtype=np.float32)
    option_indices = np.full((n_scans, max_options), -1, dtype=np.int16)
    option_mask = np.zeros((n_scans, max_options), dtype=bool)
    time_mask = np.zeros((n_scans, max_steps), dtype=bool)

    for i, item in enumerate(built):
        n_options, n_steps = item["sequence"].shape[:2]
        temporal[i, :n_options, :n_steps] = item["sequence"]
        static[i, :n_options] = item["static"]
        labels[i, :n_options] = item["labels"]
        option_indices[i, :n_options] = item["option_indices"]
        option_mask[i, :n_options] = True
        time_mask[i, :n_steps] = item["time_mask"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        schema_version=np.array(2, dtype=np.int16),
        temporal=temporal,
        static=static,
        labels=labels,
        option_indices=option_indices,
        option_mask=option_mask,
        time_mask=time_mask,
        scan_ids=np.array([item["scan_id"] for item in built]),
        topology_ids=np.array([item["topology_id"] for item in built]),
        configured_n_aps=np.array([item["n_aps"] for item in built], dtype=np.int16),
        n_hotspots=np.array([item["n_hotspots"] for item in built], dtype=np.int16),
        candidate_strata=np.array([item["candidate_stratum"] for item in built]),
        temporal_features=np.array(TEMPORAL_FEATURES),
        static_features=np.array(static_features),
        bin_ms=np.array(args.bin_ms, dtype=np.float32),
        scan_description=np.array(cfg.describe()),
    )
    print(f"wrote {n_scans} scans, {int(option_mask.sum())} options, "
          f"{max_steps} time steps x {n_temporal} features to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
