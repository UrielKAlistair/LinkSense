#!/usr/bin/env python3
"""Build the binned corpus: each option's view of a scan, one step per time bin.

See README.md in this directory for how the corpus relates to the other builders.

INPUT
  runs_dir  the run directories and _manifest.json written by run_sweep.py

PROCESS, per scan
  1. Project the recording (projection.project), or keep every channel with
     --all-channels, and find the scan's options (scans.find_options).
  2. Lay out the steps. Under the sweep each dwell is cut into whole bins of
     about --bin-ms, one step per bin, on the channel tuned during it. Under
     --all-channels each bin is four steps, one per scan channel.
  3. For each option and step: where the step lies in the window and in its
     dwell, whether its channel is the option's, the share of options on that
     channel, the channel's busy fraction, a summary of every frame in the step,
     and a summary of the frames from the option's own BSS.
  4. Fill each signal level that has no frame behind it: the noise floor when
     the radio listened and heard nothing, otherwise the last reading carried
     forward, with its age.

OUTPUT: an .npz holding temporal (scan, option, step, feature), labels, option
and time masks, and the scan and topology identifiers.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.models.temporal import TEMPORAL_SCHEMA_VERSION  # noqa: E402
from scripts.dataset.projection import (DWELL_MS, DWELL_S, RETUNE_S,  # noqa: E402
                                        SCAN_CHANNELS, TYPE_DATA, ProjectedScan,
                                        describe, project)
from scripts.dataset.scans import Option, find_options, find_scans  # noqa: E402


# One summary block, in the order _frame_summary() emits it. The last two are
# not measurements: `observed` says whether the bin held a signal reading, and
# `age` how stale the level readings in it are. Before a subject's
# first reading there is nothing to be stale, and `age` is a flat 1.0 there -
# _resolve_levels back-fills those bins from the future, so read `observed` first.
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
                   "tuned_channel_option_fraction", "cca_busy_fraction")

TEMPORAL_FEATURES = (
    PREFIX_FEATURES
    + tuple(f"channel_{n}" for n in SUMMARY_NAMES)
    + tuple(f"option_{n}" for n in SUMMARY_NAMES)
)
# Where each summary block starts, so _resolve_levels covers every block rather
# than however many were written out by hand.
BLOCK_STARTS = tuple(len(PREFIX_FEATURES) + i * SUMMARY_WIDTH for i in range(2))


def _frame_summary(frames: list[dict], rssis: list[float], listen_s: float) -> list[float]:
    """One summary block for one time bin. The levels, rssi_mean and rssi_max, are
    taken over rssis, and are NaN when it is empty.

    A bin with no reading has no level, so the level columns are NaN rather than
    a sentinel. A sentinel would sit far below anything physical, dominate the
    standardiser, and turn the time-average of the column into a count of how
    many bins held a frame. _resolve_levels() settles the NaN once the whole
    series exists, and `observed`/`age` preserve what that fill would erase.
    """
    nan = float("nan")
    levels = [float(np.mean(rssis)), max(rssis)] if rssis else [nan, nan]
    if not frames:
        return [0.0, 0.0, 0.0, *levels, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    count = len(frames)
    data = [r for r in frames if r["cat"] == TYPE_DATA]
    transmitters = {r["ta"] for r in frames if r["ta"]}
    return [
        math.log1p(count),
        math.log1p(sum(r["len"] for r in frames)),
        min(1.0, sum(r["dur"] for r in frames) / (listen_s * 1e6)) if listen_s else 0.0,
        *levels,
        math.log1p(sum(r["beacon"] for r in frames)),
        len(data) / count,
        sum(r["retry"] for r in frames) / count,
        math.log1p(float(np.mean([r["rate"] for r in data]))) if data else 0.0,
        math.log1p(len(transmitters)),
        1.0,   # observed; _resolve_levels rewrites the whole column from the
               # NaN pattern, so this value only has to be non-NaN
        0.0,   # age, filled in by _resolve_levels
    ]


def _resolve_levels(out: np.ndarray, base: int, quiet_is_floor: bool) -> None:
    """Give every bin in one summary block a level, per option.

    With quiet_is_floor, a bin without a reading gets the noise floor, with age
    0. Otherwise it gets the last reading carried forward, the way a scan cache
    holds a level until something replaces it, and age counts how long ago that
    reading was taken; before the first reading, the first is carried backwards.
    """
    n_options, n_steps, _ = out.shape
    steps = np.arange(n_steps)
    heard = ~np.isnan(out[:, :, base + LEVEL_OFFSETS[0]])

    for option in range(n_options):
        known = heard[option]
        if quiet_is_floor:
            for offset in LEVEL_OFFSETS:
                out[option, ~known, base + offset] = NOISE_FLOOR_DBM
            known = np.ones(n_steps, dtype=bool)
        valid = np.flatnonzero(known)
        last = np.where(known, steps, -1)
        np.maximum.accumulate(last, out=last)
        # before the first reading there is nothing behind to carry, so the
        # first one is carried backwards instead
        source = np.where(last >= 0, last, valid[0])
        for offset in LEVEL_OFFSETS:
            out[option, :, base + offset] = out[option, source, base + offset]
        age = np.where(last >= 0, steps - last, n_steps)
        out[option, :, base + AGE_OFFSET] = np.minimum(age / max(1, n_steps), 1.0)
    out[:, :, base + OBSERVED_OFFSET] = heard


@dataclasses.dataclass(frozen=True)
class Step:
    """One step of a scan's sequence: one time bin, heard on one channel."""

    channel: int
    time_bin: int
    dwell_position: float   # centre of the bin within its dwell, 0-1
    listen_start: float     # after any retuning
    end: float


def lay_out_steps(projected: ProjectedScan,
                  bin_ms: float) -> tuple[list[Step], int, float]:
    """The steps of a scan in order, the number of time bins, and the bin width.

    Under a sweep each dwell is cut into whole bins, so the width is the dwell
    divided by ceil(dwell / bin_ms), and each bin is one step on the channel
    tuned during it. Without a sweep each bin is one step per scan channel, and
    counts as a whole dwell of its own.
    """
    if projected.dwell_schedule is None:
        bin_s = bin_ms / 1000.0
        n_bins = max(1, int(round(projected.window / bin_s)))
        steps = [Step(channel, b, 0.5, b * bin_s, b * bin_s + bin_s)
                 for b in range(n_bins) for channel in SCAN_CHANNELS]
        return steps, n_bins, bin_s

    bins_per_dwell = max(1, math.ceil(DWELL_MS / bin_ms))
    bin_s = DWELL_S / bins_per_dwell
    n_bins = len(projected.dwell_schedule) * bins_per_dwell
    steps = []
    for b in range(n_bins):
        slot = b // bins_per_dwell
        start = projected.sweep_start + b * bin_s
        slot_start = projected.sweep_start + slot * DWELL_S
        steps.append(Step(projected.dwell_schedule[slot], b,
                          (b % bins_per_dwell + 0.5) / bins_per_dwell,
                          max(start, slot_start + RETUNE_S), start + bin_s))
    return steps, n_bins, bin_s


def scan_sequence(projected: ProjectedScan, options: list[Option],
                  bin_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (N options, T steps, F features) and the valid-step mask."""
    steps, n_bins, bin_s = lay_out_steps(projected, bin_ms)
    out = np.zeros((len(options), len(steps), len(TEMPORAL_FEATURES)), dtype=np.float32)
    time_mask = np.ones(len(steps), dtype=bool)

    step_of = {(s.time_bin, s.channel): i for i, s in enumerate(steps)}
    frames_by_step: list[list[dict]] = [[] for _ in steps]
    for frame in projected.frames:
        b = math.floor((frame["tx_start"] - projected.sweep_start) / bin_s)
        i = step_of.get((b, frame["channel"]))
        if i is not None:
            frames_by_step[i].append(frame)

    # TODO: bin edges are computed in seconds, sweep_start + b * bin_s, so a
    # chanbusy.csv millisecond starting exactly on an edge can floor into the bin
    # before it. It is placed by its midpoint instead. Computing the edges in whole
    # milliseconds, as projection.in_dwell does, would let it be placed by its start.
    shares_by_step: list[list[float]] = [[] for _ in steps]
    for row in projected.busy:
        b = math.floor(((row["start"] + row["end"]) / 2 - projected.sweep_start) / bin_s)
        i = step_of.get((b, row["channel"]))
        if i is not None:
            shares_by_step[i].append(row["busy_frac"])

    option_count_by_channel: dict[int, int] = {}
    for option in options:
        option_count_by_channel[option.channel] = option_count_by_channel.get(option.channel, 0) + 1

    for i, step in enumerate(steps):
        listen_s = max(0.0, step.end - step.listen_start)
        frames = frames_by_step[i]
        channel_summary = _frame_summary(frames, [r["rssi"] for r in frames], listen_s)
        shares = shares_by_step[i]
        cca = sum(shares) / len(shares) if shares else 0.0

        for option_index, option in enumerate(options):
            option_frames = [r for r in frames if r["bssid"] == option.mac]
            # the option's levels are its AP's signal, so they come from its beacons
            beacon_rssis = [r["rssi"] for r in option_frames if r["beacon"]]
            prefix = [
                (step.time_bin + 0.5) / n_bins,
                step.dwell_position,
                float(step.channel == option.channel),
                option_count_by_channel.get(step.channel, 0) / len(options),
                cca,
            ]
            out[option_index, i] = prefix + channel_summary + \
                _frame_summary(option_frames, beacon_rssis, listen_s)

    # The channel block's levels come from every frame on the channel the radio was
    # tuned to, so a step with none heard the noise floor. The option block's come
    # from its AP's beacons, and a step without one leaves the AP's level unknown.
    _resolve_levels(out, BLOCK_STARTS[0], quiet_is_floor=True)
    _resolve_levels(out, BLOCK_STARTS[1], quiet_is_floor=False)
    # A real check, not an assert: an unresolved NaN poisons the standardiser
    # and every metric downstream, and -O must not be able to switch it off.
    if np.isnan(out).any():
        raise ValueError("unresolved NaN in binned tensor")
    return out, time_mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None,
                        help="default data/binned.npz, or data/binned_all_channels.npz "
                             "with --all-channels")
    parser.add_argument("--bin-ms", type=float, default=None,
                        help="bin width in ms (default 10, or 100 with --all-channels); "
                             "under the sweep, narrowed so whole bins fill each dwell")
    parser.add_argument("--all-channels", action="store_true",
                        help="keep every channel for the whole window instead of "
                             "projecting onto one radio")
    args = parser.parse_args()
    bin_ms = args.bin_ms if args.bin_ms is not None else (100.0 if args.all_channels else 10.0)
    if bin_ms <= 0:
        parser.error("--bin-ms must be positive")
    out_path = args.out or Path("data/binned_all_channels.npz" if args.all_channels
                                else "data/binned.npz")
    print(f"projection: {describe(args.all_channels)}")

    scans = find_scans(args.runs_dir)
    built = []
    for number, scan in enumerate(scans, 1):
        projected = project(scan.recording, scan.scan_id, args.all_channels)
        options = find_options(scan, projected)
        if options is not None:
            sequence, time_mask = scan_sequence(projected, options, bin_ms)
            params = options[0].metadata["params"]
            built.append({
                "scan_id": scan.scan_id,
                "topology_id": scan.topology_id,
                "n_aps": int(params["n_aps"]),
                "n_hotspots": int(params["n_hotspots"]),
                "sequence": sequence,
                "time_mask": time_mask,
                "option_indices": np.array([option.index for option in options]),
                "labels": np.array([option.throughput_mbps for option in options],
                                   dtype=np.float32),
            })
        if number % 25 == 0 or number == len(scans):
            print(f"[{number}/{len(scans)}] binned scans")

    max_options = max(len(item["option_indices"]) for item in built)
    max_steps = max(item["sequence"].shape[1] for item in built)
    n_scans = len(built)
    n_temporal = len(TEMPORAL_FEATURES)

    temporal = np.zeros((n_scans, max_options, max_steps, n_temporal), dtype=np.float32)
    labels = np.zeros((n_scans, max_options), dtype=np.float32)
    option_indices = np.full((n_scans, max_options), -1, dtype=np.int16)
    option_mask = np.zeros((n_scans, max_options), dtype=bool)
    time_mask = np.zeros((n_scans, max_steps), dtype=bool)

    for i, item in enumerate(built):
        n_options, n_steps = item["sequence"].shape[:2]
        temporal[i, :n_options, :n_steps] = item["sequence"]
        labels[i, :n_options] = item["labels"]
        option_indices[i, :n_options] = item["option_indices"]
        option_mask[i, :n_options] = True
        time_mask[i, :n_steps] = item["time_mask"]

    # the width the bins actually have: under the sweep the dwell is divided into
    # a whole number of bins, so 7 ms of a 110 ms dwell is 6.875
    actual_bin_ms = (bin_ms if args.all_channels
                     else DWELL_MS / max(1, math.ceil(DWELL_MS / bin_ms)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        schema_version=np.array(TEMPORAL_SCHEMA_VERSION, dtype=np.int16),
        temporal=temporal,
        labels=labels,
        option_indices=option_indices,
        option_mask=option_mask,
        time_mask=time_mask,
        scan_ids=np.array([item["scan_id"] for item in built]),
        topology_ids=np.array([item["topology_id"] for item in built]),
        configured_n_aps=np.array([item["n_aps"] for item in built], dtype=np.int16),
        n_hotspots=np.array([item["n_hotspots"] for item in built], dtype=np.int16),
        temporal_features=np.array(TEMPORAL_FEATURES),
        bin_ms=np.array(actual_bin_ms, dtype=np.float32),
        scan_description=np.array(describe(args.all_channels)),
    )
    print(f"wrote {n_scans} scans, {int(option_mask.sum())} options, "
          f"{max_steps} steps x {n_temporal} features to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
