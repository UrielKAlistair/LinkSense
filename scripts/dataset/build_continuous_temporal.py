#!/usr/bin/env python3
"""Build the full-observation temporal corpus: every channel, every bin.

build_temporal_dataset.py models a real single-radio client, which can only
listen to one channel at a time. That carves each AP's history down to a
single 110 ms dwell - about one beacon - so a sequence model has almost
nothing to read. This script removes that restriction and keeps the whole
parallel-scanner superset the simulator recorded: each option gets a
*continuous* series on its own channel across the entire pre-association
window.

This is deliberately optimistic about hardware. A commodity station cannot
observe four channels at once. The point is to establish an upper bound: if a
sequence model cannot beat aggregate features even with continuous
observation, then scan dwell was never the limitation and the trajectory
genuinely carries nothing. If it can, the deficit is a scan-policy artifact
and the roaming setting (where the serving channel *is* observed
continuously) becomes the right venue.

RSSI degradation is still applied - that is measurement realism, not scan
policy, and removing it would confound the comparison.

Run:
  python scripts/build_continuous_temporal.py data/combined_runs \
      data/combined_dataset.csv --out data/combined_continuous.npz --bin-ms 100
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
from scripts.dataset.build_dataset import (ScanConfig, _rng, apply_rssi_realism,  # noqa: E402
                                   read_chanbusy, read_observation)
from scripts.dataset.build_temporal_dataset import (AGE_OFFSET, LEVEL_OFFSETS,  # noqa: E402
                                            NOISE_FLOOR_DBM, OBSERVED_OFFSET,
                                            SUMMARY_NAMES, SUMMARY_WIDTH,
                                            _fill_levels, _frame_summary,
                                            _overlap)


def reference_dir(runs_dirs: list[Path], topology_id: str, seed_key: str) -> Path:
    """Locate a run directory across one or more sweep output roots.

    combined_dataset.csv is stitched from several sweeps (development "g*"
    topologies, extension "x*"), so the runs for one dataset can live under
    different roots. Prefixes are disjoint per sweep, so the first match is
    unambiguous.
    """
    prefix = f"{topology_id}__{seed_key}__"
    for runs_dir in runs_dirs:
        exact = runs_dir / f"{prefix}ap0"
        if (exact / "observation.csv").exists():
            return exact
        refs = [d for d in runs_dir.iterdir()
                if d.is_dir() and d.name.startswith(prefix)
                and (d / "observation.csv").exists()]
        if refs:
            return sorted(refs)[0]
    raise FileNotFoundError(
        f"no observation for {topology_id}/{seed_key} under {[str(d) for d in runs_dirs]}")

PREFIX_FEATURES = ("time_fraction", "own_cca_busy_fraction")

TEMPORAL_FEATURES = (
    PREFIX_FEATURES
    + tuple(f"own_channel_{n}" for n in SUMMARY_NAMES)
    + tuple(f"option_{n}" for n in SUMMARY_NAMES)
    + tuple(f"all_channel_{n}" for n in SUMMARY_NAMES)
    # Not a share: a ratio of log1p counts, monotone in the true share but
    # equal to 1.0 whenever one channel carries all of the traffic, whatever
    # the volume. Named for what it is rather than what it approximates.
    + ("n_channels_busy", "own_channel_log_frame_ratio")
)
BLOCK_STARTS = tuple(len(PREFIX_FEATURES) + i * SUMMARY_WIDTH for i in range(3))


def continuous_group(rows: list[dict], busy: list[dict], option_meta: list[dict],
                     t_start: float, t_end: float, bin_ms: float
                     ) -> tuple[np.ndarray, np.ndarray]:
    """(N options, T bins, F features) over the whole window, all channels live."""
    bin_s = bin_ms / 1000.0
    n_steps = max(1, int(round((t_end - t_start) / bin_s)))
    out = np.zeros((len(option_meta), n_steps, len(TEMPORAL_FEATURES)), dtype=np.float32)

    channels = sorted({ap["channel"] for ap in option_meta})
    # Bin every frame once, by the channel it was heard on.
    by_channel_step: dict[int, list[list[dict]]] = {
        ch: [[] for _ in range(n_steps)] for ch in channels}
    all_step: list[list[dict]] = [[] for _ in range(n_steps)]
    for frame in rows:
        channel = (frame["freq"] - 5000) // 5
        start = frame["t"] - frame["dur"] / 1e6
        step = int((start - t_start) / bin_s)
        if not (0 <= step < n_steps):
            continue
        all_step[step].append(frame)
        if channel in by_channel_step:
            by_channel_step[channel][step].append(frame)

    busy_by_channel: dict[int, list[dict]] = {ch: [] for ch in channels}
    for event in busy:
        if event["channel"] in busy_by_channel:
            busy_by_channel[event["channel"]].append(event)

    # Channel-level quantities are shared by every option on that channel, so
    # compute them once per (channel, step) rather than once per option.
    channel_summary: dict[int, list[list[float]]] = {}
    channel_cca: dict[int, list[float]] = {}
    for ch in channels:
        summaries, ccas = [], []
        for step in range(n_steps):
            start = t_start + step * bin_s
            end = start + bin_s
            summaries.append(_frame_summary(by_channel_step[ch][step], bin_s))
            busy_s = sum(_overlap(e["start"], e["end"], start, end) * e["busy_frac"]
                         for e in busy_by_channel[ch])
            ccas.append(min(1.0, busy_s / bin_s))
        channel_summary[ch] = summaries
        channel_cca[ch] = ccas

    all_summary = [_frame_summary(all_step[step], bin_s) for step in range(n_steps)]
    busy_count = [sum(1 for ch in channels if channel_cca[ch][step] > 0.01)
                  for step in range(n_steps)]

    option_frames_step: list[list[list[dict]]] = []
    for ap in option_meta:
        per_step = [[] for _ in range(n_steps)]
        for step in range(n_steps):
            for frame in by_channel_step[ap["channel"]][step]:
                if frame["bssid"] == ap["mac"]:
                    per_step[step].append(frame)
        option_frames_step.append(per_step)

    for option_index, ap in enumerate(option_meta):
        own = channel_summary[ap["channel"]]
        cca = channel_cca[ap["channel"]]
        for step in range(n_steps):
            # index 0 of a summary is log1p(frame count); the ratio below is a
            # share of log-counts, which is bounded and monotone in the real
            # share, and avoids dividing by an empty all-channel bin.
            ratio = own[step][0] / all_summary[step][0] if all_summary[step][0] > 0 else 0.0
            out[option_index, step] = (
                [(step + 0.5) / n_steps, cca[step]]
                + own[step]
                + _frame_summary(option_frames_step[option_index][step], bin_s)
                + all_summary[step]
                + [busy_count[step] / max(1, len(channels)), ratio]
            )
    _fill_levels(out, BLOCK_STARTS)
    assert not np.isnan(out).any(), "unresolved NaN in continuous tensor"
    return out, np.ones(n_steps, dtype=bool)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", type=Path, nargs="+")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/continuous.npz"))
    parser.add_argument("--bin-ms", type=float, default=100.0,
                        help="time-bin width; beacons arrive every 102.4 ms")
    parser.add_argument("--start-s", type=float, default=0.0,
                        help="observation start; the window ends at feature_window_end")
    args = parser.parse_args()
    if args.bin_ms <= 0:
        parser.error("--bin-ms must be positive")

    frame = impute_features(load_dataset(args.dataset))
    value = {c: frame[c].dropna().unique()[0] for c in (
        "meta_rssi_noise_db", "meta_rssi_noise_model", "meta_rssi_bias_db",
        "meta_rssi_quant_db", "meta_scan_seed")}
    # Only the RSSI-realism half of ScanConfig is used here; sweep=False means
    # no dwell schedule is applied and every channel stays observable.
    cfg = ScanConfig(
        sweep=False, mode="passive",
        rssi_noise_db=float(value["meta_rssi_noise_db"]),
        rssi_noise_model=str(value["meta_rssi_noise_model"]),
        rssi_bias_db=float(value["meta_rssi_bias_db"]),
        rssi_quant_db=float(value["meta_rssi_quant_db"]),
        seed=int(value["meta_scan_seed"]))

    static_features = feature_columns(frame)
    built = []
    total = frame["group_id"].nunique()
    for number, (group_id, group) in enumerate(frame.groupby("group_id", sort=True), 1):
        group = group.sort_values("ap_index")
        topology_id = str(group["topology_id"].iloc[0])
        seed_key = group_id.rsplit("__", 1)[-1] if "__s" in group_id else "s00"
        ref_dir = reference_dir(args.runs_dir, topology_id, seed_key)
        meta = json.loads((ref_dir / "metadata.json").read_text())
        window_end = float(meta["params"]["feature_window_end"])

        ap_by_index = {int(ap["index"]): {
            "index": int(ap["index"]), "mac": ap["mac"].lower(),
            "channel": int(ap["channel"])} for ap in meta["aps"]}

        rows = read_observation(ref_dir / "observation.csv")
        apply_rssi_realism(rows, cfg, _rng(cfg, group_id, "rssi"))

        option_indices = group["ap_index"].astype(int).to_numpy()
        option_meta = [ap_by_index[index] for index in option_indices]
        sequence, time_mask = continuous_group(
            rows, read_chanbusy(ref_dir / "chanbusy.csv"), option_meta,
            args.start_s, window_end, args.bin_ms)

        built.append({
            "group_id": group_id, "topology_id": topology_id,
            "n_aps": int(group["gt_n_aps"].iloc[0]),
            "n_hotspots": int(group["gt_n_hotspots"].iloc[0]),
            "candidate_stratum": str(group["gt_candidate_stratum"].iloc[0]),
            "sequence": sequence, "time_mask": time_mask,
            "static": group[static_features].to_numpy(dtype=np.float32),
            "option_indices": option_indices,
            "labels": group["label_throughput_mbps"].to_numpy(dtype=np.float32),
        })
        if number % 100 == 0 or number == total:
            print(f"[{number}/{total}] continuous groups")

    max_options = max(len(i["option_indices"]) for i in built)
    max_steps = max(i["sequence"].shape[1] for i in built)
    n_groups = len(built)

    temporal = np.zeros((n_groups, max_options, max_steps, len(TEMPORAL_FEATURES)),
                        dtype=np.float32)
    static = np.zeros((n_groups, max_options, len(static_features)), dtype=np.float32)
    labels = np.zeros((n_groups, max_options), dtype=np.float32)
    option_indices = np.full((n_groups, max_options), -1, dtype=np.int16)
    option_mask = np.zeros((n_groups, max_options), dtype=bool)
    time_mask = np.zeros((n_groups, max_steps), dtype=bool)
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
        args.out, schema_version=np.array(2, dtype=np.int16),
        temporal=temporal, static=static, labels=labels,
        option_indices=option_indices, option_mask=option_mask, time_mask=time_mask,
        group_ids=np.array([i["group_id"] for i in built]),
        topology_ids=np.array([i["topology_id"] for i in built]),
        configured_n_aps=np.array([i["n_aps"] for i in built], dtype=np.int16),
        n_hotspots=np.array([i["n_hotspots"] for i in built], dtype=np.int16),
        candidate_strata=np.array([i["candidate_stratum"] for i in built]),
        temporal_features=np.array(TEMPORAL_FEATURES),
        static_features=np.array(static_features),
        bin_ms=np.array(args.bin_ms, dtype=np.float32),
        scan_description=np.array(
            f"continuous all-channel observation, {args.start_s}-window_end s, "
            f"{args.bin_ms} ms bins; RSSI realism applied, no dwell restriction"))
    print(f"wrote {n_groups} groups, {int(option_mask.sum())} options, "
          f"{max_steps} steps x {len(TEMPORAL_FEATURES)} features to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
