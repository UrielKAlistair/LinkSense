#!/usr/bin/env python3
"""Project a scan's recording onto what one client radio could have received.

INPUT, from the run directory that recorded the scan
  observation.csv  every frame the simulator's scanner radios decoded, one
                   radio per channel, each listening for the whole window
  chanbusy.csv     for every 1 ms, the share each scanner radio sensed busy
  metadata.json    read only for the length of the listening window

PROCESS, in memory, in project()
  1. Round every frame's RSSI to a whole dBm, the resolution a device reports.
  2. Schedule a sweep: one radio visits channels 36, 40, 44 and 48 in a random
     order, 110 ms per visit, for as many whole passes as the window holds,
     the last visit ending at the window end.
  3. Keep a frame, and a chanbusy.csv millisecond, only if the radio was on its
     channel, and done retuning, from its start to its end.

  With all_channels steps 2 and 3 are skipped and every channel is kept for the
  whole window. The visit order is drawn from SEED and the scan ID, so every
  builder computes the same projection.

OUTPUT
  ProjectedScan: the kept frames, the kept chanbusy.csv milliseconds and the
  dwell schedule.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import random
from pathlib import Path

TYPE_MGMT, TYPE_CTRL, TYPE_DATA = 0, 1, 2

# The channels the client radio visits, whether or not an AP is on them.
# Taking them from the deployment instead would tell the model how many channels
# are occupied. Must match kApChannels in sim/my-wifi-test.cc.
SCAN_CHANNELS = (36, 40, 44, 48)

# One visit to a channel, and the deaf time at its start while the radio retunes.
DWELL_MS = 110.0
RETUNE_MS = 0.15
DWELL_S = DWELL_MS / 1000.0
RETUNE_S = RETUNE_MS / 1000.0
LISTEN_PER_DWELL_S = (DWELL_MS - RETUNE_MS) / 1000.0

# Combined with each scan's ID to seed its visit order.
SEED = 20250817


def channel_of_freq(freq_mhz: int) -> int:
    return (freq_mhz - 5000) // 5


def describe(all_channels: bool) -> str:
    """The projection in words, stored with each corpus."""
    if all_channels:
        return "all channels observed in parallel for the full window"
    return (f"single-radio passive sweep: {DWELL_MS:g} ms dwell ({RETUNE_MS:g} ms "
            "retune), window-filling passes, random order, aligned to window end")


def sweep_rng(scan_id: str) -> random.Random:
    """The generator for one scan's visit order, seeded from a digest of SEED and
    the scan ID.

    hash() is randomised per process, so it would give each worker different
    draws.
    """
    d = hashlib.blake2b(f"{SEED}:{scan_id}:sweep".encode(), digest_size=8)
    return random.Random(int.from_bytes(d.digest(), "big"))


# ---------------------------------------------------------------------------
# Reading a recording
# ---------------------------------------------------------------------------

def read_observation(path: Path) -> list[dict]:
    """observation.csv as one dict per decoded frame.

    tx_start and tx_end are the span of the transmission carrying the frame, in
    seconds from the start of listening; every frame of an aggregate has the
    same span. rssi is rounded to a whole dBm. bssid and ta are lower case, or
    None when the frame has none. cat is TYPE_MGMT, TYPE_CTRL or TYPE_DATA. dur
    is the frame's own airtime in microseconds, len its size in bytes, rate its
    PHY rate in Mbit/s.
    """
    rows = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "tx_start": float(r["tx_start"]),
                "tx_end": float(r["tx_end"]),
                "channel": channel_of_freq(int(r["freq_mhz"])),
                "bssid": r["bssid"].lower() or None,
                "ta": r["ta"].lower() or None,
                "cat": int(r["cat"]),
                "beacon": r["is_beacon"] == "1",
                "retry": r["retry"] == "1",
                "len": int(r["len"]),
                "rssi": float(round(float(r["signal_dbm"]))),
                "dur": float(r["duration_us"]),
                "rate": float(r["rate_mbps"]),
            })
    return rows


def read_chanbusy(path: Path) -> list[dict]:
    """chanbusy.csv as dicts with start, end, channel and busy_frac.

    Each row is one 1 ms bucket of one channel. busy_frac is the share of it in
    which that channel's scanner radio was transmitting, receiving or sensing
    energy, whether or not a frame decoded.
    """
    with path.open(newline="") as f:
        return [{"start": float(r["start"]), "end": float(r["end"]),
                 "channel": int(r["channel"]), "busy_frac": float(r["busy_frac"])}
                for r in csv.DictReader(f)]


# ---------------------------------------------------------------------------
# The projection
# ---------------------------------------------------------------------------

def sweep_schedule(window: float, rng: random.Random) -> tuple[list[int], float]:
    """The channel visited in each dwell, and when the first dwell starts, in
    milliseconds.

    As many whole passes over SCAN_CHANNELS as fit in the window, every pass in
    one order shuffled per scan, the last dwell ending at the window end, when
    the client joins. A 6 s window holds 13 passes, 52 dwells, from 280 ms.
    """
    order = list(SCAN_CHANNELS)
    rng.shuffle(order)
    window_ms = window * 1000.0
    n_passes = int(window_ms // (DWELL_MS * len(order)))
    n_dwells = n_passes * len(order)
    return [order[i % len(order)] for i in range(n_dwells)], window_ms - n_dwells * DWELL_MS


def in_dwell(channel: int, start: float, end: float, dwell_schedule: list[int],
             sweep_start_ms: float) -> bool:
    """Whether the radio was tuned to channel, done retuning, from start to end, in
    seconds.

    A frame's span is that of its transmission, which every frame of an
    aggregate shares, so an aggregate is kept or dropped whole.
    """
    # floor, not int(): int() truncates toward zero, so a span starting just
    # before the sweep would land in dwell 0 instead of being rejected
    i = math.floor((start * 1000.0 - sweep_start_ms) / DWELL_MS)
    if not (0 <= i < len(dwell_schedule)) or channel != dwell_schedule[i]:
        return False
    # The dwell's edges are whole milliseconds, divided by 1000 only here, so an
    # edge is the same float as that millisecond read from chanbusy.csv.
    dwell_start_ms = sweep_start_ms + i * DWELL_MS
    return (start >= (dwell_start_ms + RETUNE_MS) / 1000.0
            and end <= (dwell_start_ms + DWELL_MS) / 1000.0)


@dataclasses.dataclass
class ProjectedScan:
    """One scan as the client radio received it."""

    window: float                     # length of the listening window, seconds
    frames: list[dict]                # read_observation() rows kept, in file order
    busy: list[dict]                  # read_chanbusy() rows kept, in file order
    dwell_schedule: list[int] | None  # each dwell's channel in turn; None with all_channels
    sweep_start: float                # when the first dwell starts

    @property
    def listen_s(self) -> float:
        """Seconds spent receiving on each scan channel, the same for all of them."""
        if self.dwell_schedule is None:
            return self.window
        return len(self.dwell_schedule) // len(SCAN_CHANNELS) * LISTEN_PER_DWELL_S


def project(run_dir: Path, scan_id: str, all_channels: bool) -> ProjectedScan:
    """Read the scan recorded in run_dir and project it onto one radio, or keep
    every channel for the whole window when all_channels is set."""
    metadata = json.loads((run_dir / "metadata.json").read_text())
    window = float(metadata["params"]["feature_window_end"])
    frames = read_observation(run_dir / "observation.csv")
    busy = read_chanbusy(run_dir / "chanbusy.csv")
    if all_channels:
        return ProjectedScan(window, frames, busy, None, 0.0)

    dwell_schedule, sweep_start_ms = sweep_schedule(window, sweep_rng(scan_id))
    frames = [r for r in frames if in_dwell(r["channel"], r["tx_start"], r["tx_end"],
                                            dwell_schedule, sweep_start_ms)]
    busy = [e for e in busy if in_dwell(e["channel"], e["start"], e["end"],
                                        dwell_schedule, sweep_start_ms)]
    return ProjectedScan(window, frames, busy, dwell_schedule, sweep_start_ms / 1000.0)

