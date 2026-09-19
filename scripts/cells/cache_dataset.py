#!/usr/bin/env python3
"""Cache each scan as the cells its models read.

INPUT
  All simulated observation data: runs_dir and _manifest.json written by run_sweep.py

OUTPUT
  <scan_id>.npz per scan, and _index.npz naming the scans, their topologies and
  their deployment sizes, so the trainer can split without opening any of them.
  Output npz files are directly usable for training.

PROCESS, per scan
  1. Project the recording onto one radio (common.projection), and cut what it
     heard into its dwells.
  2. Put every frame of every dwell into one frame table, tagged with its dwell
     and with the AP number of the valid AP whose BSSID it carries.
  3. For every dwell, emit one "channel" cell, and one for each valid AP on that channel.
     Every channel is visited num_passes times, as is every valid AP
     (an AP whose beacon was decoded at least once in the projected recording).
     A scan thus has num_passes x (num_channels + num_valid_APs) cells.
     Each cell is built with its frames, aggregate statistics over all of them, and, for channel
     cells, that dwell's 109 carrier-sense milliseconds.
  4. Summarise each valid AP over the whole window into a descriptor.


Each frame is stored once, in the order the radio heard it. A cell stores no
frames of its own: Cell.find_frames reads them back from those tags, so an
AP cell is by construction a subset of its dwell's channel cell. Frames are cached whole:
capping how many a cell hands the model is a training decision - a random
subset while training, an even stride at evaluation - and cannot be baked in
here.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import functools
import math
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.common.projection import (DWELL_MS, DWELL_S,  # noqa: E402
                                       TYPE_DATA, TYPE_MGMT, ProjectedScan,
                                       describe, dwell_bounds, dwell_of, project)
from scripts.common.parse_scans import Scan, ValidAP, find_scans, find_valid_aps  # noqa: E402

CHANNEL_CELL, AP_CELL = 0, 1
NO_AP = -1
N_CATEGORIES = 3

# Milliseconds of carrier sense a dwell yields: the projection drops the first,
# which the radio spent retuning. 
CCA_SAMPLES = int(DWELL_MS) - 1

# Signal level recorded where there is no level to report: below the simulated
# noise floor of about -94 dBm, so it reads as weaker than anything heard.
MISSING_RSSI_DBM = -100.0

FRAME_FEATURES = (
    "offset_in_dwell",      # where in the 110 ms the transmission started, 0-1
    "gap_log1p",            # idle microseconds before it on this channel
    "rssi_dbm",
    "airtime_log1p",        # this frame's own airtime, not its transmission's
    "length_log1p",
    "rate_log1p",
    "is_beacon",
    "is_retry",
    "from_ap",              # the BSSID itself sent it, not one of its stations
    "of_valid_ap",          # carries a valid AP's BSSID
)

CELL_AGGREGATES = (
    "frames_log1p",
    "bytes_log1p",
    "airtime_fraction",       # the frames' own airtimes summed, over the time listened
    "busy_fraction",          # time with a decoded transmission on air, overlaps counted once
    "data_fraction",
    "mgmt_fraction",
    "retry_fraction",
    "transmitters_log1p",     # distinct transmitters, the AP itself left out on an AP cell
    "beacons_log1p",          # the AP's beacons; 0 on a channel cell
    "beacon_rssi_mean",       # their level; MISSING_RSSI_DBM on a channel cell
    "rssi_mean",              # over every frame, its stations' as well as the AP's
    "rssi_max",
    "data_rate_log1p",        # mean PHY rate of the data frames
    "longest_gap_fraction",   # longest stretch with nothing decoded on air, over the time listened
    "cca_mean",               # carrier sense over the dwell; 0 on an AP cell
    "cca_max",
)

DESCRIPTOR_FEATURES = (
    "beacon_rssi_mean",       # level of its beacons over the whole window
    "beacon_rssi_std",
    "beacon_rssi_min",
    "beacon_rssi_max",
    "beacon_rssi_last",       # level of the latest beacon heard
    "beacons_log1p",
    "bss_airtime_fraction",   # its BSS's summed airtime, over the time listened on its channel
    "bss_transmitters_log1p", # distinct stations heard sending in its BSS
    "cochannel_valid_aps",    # other valid APs on its channel
    "n_valid_aps",
)


def build_scan(scan: Scan, out_dir: Path) -> dict | None:
    """Write one scan's .npz cache, and return what the index and the closing
    summary need to know about it.

    None when the scan offers no choice, which find_valid_aps already reports.
    """
    # 1. Project onto one radio, and cut what it heard into dwells.
    projected = project(scan.recording, scan.scan_id, all_channels=False)
    valid_aps = find_valid_aps(scan, projected)
    if valid_aps is None:
        return None
    dwells = split_into_dwells(projected)

    # 2. Put every frame into the frame table.
    table = FrameTable(dwells, valid_aps)

    # 3. Emit the cells, each with its frames, aggregate statistics and carrier sense.
    cells = emit_cells(table, dwells, valid_aps)

    # 4. Summarise each valid AP over the whole window.
    descriptors = DescriptorTable(table, dwells, valid_aps)

    np.savez(
        out_dir / f"{scan.scan_id}.npz",
        frames=table.features,
        frame_categories=table.categories,
        frame_dwell=table.dwell_numbers,
        frame_ap=table.ap_numbers,
        cell_dwell=np.array([cell.dwell_number for cell in cells], dtype=np.uint8),
        cell_kind=np.array([cell.kind for cell in cells], dtype=np.uint8),
        cell_channel=np.array([cell.channel for cell in cells], dtype=np.uint8),
        cell_ap=np.array([cell.ap_number for cell in cells], dtype=np.int8),
        cell_aggregates=np.array([list(cell.aggregates.values()) for cell in cells],
                                 dtype=np.float32),
        dwell_channel=np.array([dwell.channel for dwell in dwells], dtype=np.uint8),
        # one channel cell per dwell, in dwell order, so row d is dwell d's
        cca_samples=np.stack([cell.carrier_sense for cell in cells
                              if cell.kind == CHANNEL_CELL]),
        descriptors=descriptors.features,
        labels=np.array([ap.throughput_mbps for ap in valid_aps], np.float32),
        ap_indices=np.array([ap.index for ap in valid_aps], np.int16),
        ap_channels=np.array([ap.channel for ap in valid_aps], np.uint8),
    )
    params = valid_aps[0].metadata["params"]
    return {
        "scan_id": scan.scan_id,
        "topology_id": scan.topology_id,
        "n_valid_aps": len(valid_aps),
        "n_aps": int(params["n_aps"]),
        "n_hotspots": int(params["n_hotspots"]),
        "n_cells": len(cells),
        "n_frames": len(table.heard),
        "feature_names": (table.feature_names, list(cells[0].aggregates),
                          descriptors.feature_names),
    }


# ---------------------------------------------------------------------------
# 1. Cut the projected recording into dwells
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Dwell:
    """What the radio heard while tuned to one channel for 110 ms."""

    number: int               # its place in the sweep, 0 to 51
    channel: int
    start: float              # when the radio tuned to the channel, in seconds
    listen_start: float       # when it finished retuning and began to listen
    end: float
    frames: list[dict]        # every frame it decoded, in the order heard
    cca: list[float]          # carrier sense per millisecond; none if nobody uses the channel


def split_into_dwells(projected: ProjectedScan) -> list[Dwell]:
    """The projected recording as its dwells, each holding what it heard.

    The projection kept only what lies wholly inside a dwell, so the dwell a
    frame or a carrier-sense millisecond started in is the one that heard it.
    """
    sweep_start_ms = projected.sweep_start_ms
    dwells = [Dwell(index, channel, *dwell_bounds(index, sweep_start_ms), [], [])
              for index, channel in enumerate(projected.dwell_schedule)]
    for frame in projected.frames:
        dwells[dwell_of(frame["tx_start"], sweep_start_ms)].frames.append(frame)
    for row in projected.busy:
        dwells[dwell_of(row["start"], sweep_start_ms)].cca.append(row["busy_frac"])
    for dwell in dwells:
        dwell.frames.sort(key=lambda f: f["tx_start"])
    return dwells


# ---------------------------------------------------------------------------
# 2. Put every frame into a frame table
# ---------------------------------------------------------------------------

# It is wasteful to repeatedly store frames within cells as an array, so we simply
# generate a "frame table", a python object containing all the frames in the observation
# set, but tagged with its AP number and dwell. The Cell Objects can pick out which
# frames lay in them by simply filtering on these tags.

class FrameTable:
    """Every frame of the scan once, in the order heard, built dwell by dwell."""

    features: np.ndarray      # its FRAME_FEATURES, one row per frame
    feature_names: list[str]  # the name of each column of features
    categories: np.ndarray    # management, control or data (It's a categorical feature, and is stored separately)
    dwell_numbers: np.ndarray # tag: the dwell that heard each frame
    ap_numbers: np.ndarray    # tag: the AP number whose BSSID each frame carries, or NO_AP
    heard: list[dict]         # the frames as read, which the aggregates need; not cached

    def __init__(self, dwells: list[Dwell], valid_aps: list[ValidAP]):
        ap_number_of = {ap.mac: number for number, ap in enumerate(valid_aps)}
        valid_bssids = set(ap_number_of)
        rows, dwell_numbers = [], []
        for dwell in dwells:
            rows += self.frame_features(dwell, valid_bssids)
            dwell_numbers += [dwell.number] * len(dwell.frames)
        self.heard = [frame for dwell in dwells for frame in dwell.frames]
        self.features = np.array([list(row.values()) for row in rows], dtype=np.float32)
        self.feature_names = list(rows[0]) if rows else []
        self.categories = np.array([frame["cat"] for frame in self.heard], dtype=np.uint8)
        self.dwell_numbers = np.array(dwell_numbers, dtype=np.uint8)
        self.ap_numbers = np.array([ap_number_of.get(f["bssid"], NO_AP) for f in self.heard],
                                   dtype=np.int8)

    @staticmethod
    def frame_features(dwell: Dwell, valid_bssids: set[str]) -> list[dict[str, float]]:
        """FRAME_FEATURES of each frame of one dwell, by name, in the order heard.

        The gap is the idle time before the frame: from the end of the latest
        transmission already heard on this channel, or, before the first, from
        when the radio began listening. A frame that starts before that end -
        another subframe of the same aggregate, or an overlapping reception -
        gets 0.
        """
        rows = []
        last_end = dwell.listen_start
        for frame in dwell.frames:
            rows.append({
                "offset_in_dwell": (frame["tx_start"] - dwell.start) / DWELL_S,
                "gap_log1p": math.log1p(max(frame["tx_start"] - last_end, 0.0) * 1e6),
                "rssi_dbm": frame["rssi"],
                "airtime_log1p": math.log1p(frame["dur"]),
                "length_log1p": math.log1p(frame["len"]),
                "rate_log1p": math.log1p(frame["rate"]),
                "is_beacon": float(frame["beacon"]),
                "is_retry": float(frame["retry"]),
                "from_ap": float(frame["ta"] is not None and frame["ta"] == frame["bssid"]),
                "of_valid_ap": float(frame["bssid"] in valid_bssids),
            })
            last_end = max(last_end, frame["tx_end"])
        return rows


# ---------------------------------------------------------------------------
# 3. Emit the cells, each with its frames, aggregate statistics and carrier sense
# ---------------------------------------------------------------------------

# Next, we emit the Cells, each built in one go: pointers to its respective
# FrameTable rows, its aggregate data, and, for Channel Cells, their CCABusy
# values.

def emit_cells(table: FrameTable, dwells: list[Dwell],
               valid_aps: list[ValidAP]) -> list[Cell]:
    """One channel cell for every dwell, and one AP cell for every valid AP on
    that dwell's channel, in dwell order.

    An AP cell is emitted whether or not its AP was heard in that dwell: the
    radio was listening for it, so its silence is a reading.
    """
    cells = []
    for dwell in dwells:
        cells.append(Cell(table, dwell))
        cells += [Cell(table, dwell, number, ap.mac)
                  for number, ap in enumerate(valid_aps)
                  if ap.channel == dwell.channel]
    return cells


class Cell:
    """One cell: covers a dwell, and one AP/channel.
    It holds aggregate statistics over the contents of the cell, and if it's a 
    Channel Cell, it holds CCABusy data as well. It does not hold individual frames,
    but instead a filter map for FrameTable to avoid redundant storage."""

    dwell_number: int         # which dwell, 0 to 51
    kind: int                 # CHANNEL_CELL or AP_CELL
    channel: int              # the channel the radio was tuned to
    ap_number: int            # its AP number; NO_AP on a channel cell
    frames: np.ndarray        # its rows of the frame table
    aggregates: dict[str, float]  # its CELL_AGGREGATES, by name
    carrier_sense: np.ndarray | None   # CCA_SAMPLES ms, on a channel cell only

    def __init__(self, table: FrameTable, dwell: Dwell,
                 ap_number: int = NO_AP, ap_mac: str | None = None):
        """The channel cell of `dwell`, or, given an AP, that AP's cell in it.
        """
        self.dwell_number = dwell.number
        self.kind = CHANNEL_CELL if ap_number == NO_AP else AP_CELL
        self.channel = dwell.channel
        self.ap_number = ap_number
        self.frames = self.find_frames(table)
        self.carrier_sense = self.pad_carrier_sense(dwell) if self.kind == CHANNEL_CELL else None
        self.aggregates = self.compute_aggregates(table, dwell, ap_mac)

    def find_frames(self, table: FrameTable) -> np.ndarray:
        """This cell's rows of the frame table, in the order heard.

        A channel cell holds every frame of its dwell; an AP cell holds those of
        them that carry its AP's BSSID.
        """
        mine = table.dwell_numbers == self.dwell_number
        if self.ap_number != NO_AP:
            mine &= table.ap_numbers == self.ap_number
        return np.flatnonzero(mine)

    @staticmethod
    def pad_carrier_sense(dwell: Dwell) -> np.ndarray:
        """A dwell's CCA_SAMPLES milliseconds of carrier sense, left at zero on a
        channel nobody uses."""
        samples = np.zeros(CCA_SAMPLES, dtype=np.float32)
        samples[:len(dwell.cca)] = dwell.cca
        return samples

    def compute_aggregates(self, table: FrameTable, dwell: Dwell,
                           ap_mac: str | None) -> dict[str, float]:
        """One row of CELL_AGGREGATES done over all frames of this cell.

        ap_mac is set on an AP cell, whose transmitter count estimates how many
        stations that BSS has, so the AP's own transmissions are left out of
        it. On an AP cell the gaps are the BSS's silences, which other BSSs may
        have filled, and rssi_mean and rssi_max cover its stations' frames as
        well as the AP's; the AP's own level is beacon_rssi_mean. Carrier sense
        belongs to the channel, so its columns stay zero on AP cells; the beacon
        columns belong to an AP, so a channel cell leaves them empty.
        """
        frames = [table.heard[i] for i in self.frames]
        listen_s = dwell.end - dwell.listen_start
        busy, gaps = self.air_time(frames, dwell.listen_start, dwell.end)
        count = len(frames) or 1
        rssis = [f["rssi"] for f in frames]
        beacons = [f["rssi"] for f in frames if f["beacon"]] if self.kind == AP_CELL else []
        data_rates = [f["rate"] for f in frames if f["cat"] == TYPE_DATA]
        stations = {f["ta"] for f in frames if f["ta"] and f["ta"] != ap_mac}
        cca = dwell.cca if self.kind == CHANNEL_CELL else []
        return {
            "frames_log1p": math.log1p(len(frames)),
            "bytes_log1p": math.log1p(sum(f["len"] for f in frames)),
            "airtime_fraction": sum(f["dur"] for f in frames) / (listen_s * 1e6),
            "busy_fraction": busy / listen_s,
            "data_fraction": sum(f["cat"] == TYPE_DATA for f in frames) / count,
            "mgmt_fraction": sum(f["cat"] == TYPE_MGMT for f in frames) / count,
            "retry_fraction": sum(f["retry"] for f in frames) / count,
            "transmitters_log1p": math.log1p(len(stations)),
            "beacons_log1p": math.log1p(len(beacons)),
            "beacon_rssi_mean": statistics.fmean(beacons) if beacons else MISSING_RSSI_DBM,
            "rssi_mean": statistics.fmean(rssis) if rssis else MISSING_RSSI_DBM,
            "rssi_max": max(rssis, default=MISSING_RSSI_DBM),
            "data_rate_log1p": math.log1p(statistics.fmean(data_rates)) if data_rates else 0.0,
            "longest_gap_fraction": max(gaps) / listen_s,
            "cca_mean": statistics.fmean(cca) if cca else 0.0,
            "cca_max": max(cca, default=0.0),
        }

    @staticmethod
    def air_time(frames: list[dict], start: float, end: float) -> tuple[float, list[float]]:
        """Seconds of [start, end] with something on air, and the gaps between.
        """
        busy, gaps, cursor = 0.0, [], start
        for span_start, span_end in sorted((f["tx_start"], f["tx_end"]) for f in frames):
            span_start, span_end = max(span_start, start), min(span_end, end)
            if span_end <= cursor:
                continue
            if span_start > cursor:
                gaps.append(span_start - cursor)
            busy += span_end - max(span_start, cursor)
            cursor = span_end
        gaps.append(end - cursor)
        return busy, gaps


# ---------------------------------------------------------------------------
# 4. Summarise each valid AP over the whole window
# ---------------------------------------------------------------------------

class DescriptorTable:
    """Every valid AP of the scan once, in AP-number order, each summarised over
    the frames of the frame table that carry its BSSID."""

    features: np.ndarray      # its DESCRIPTOR_FEATURES, one row per valid AP
    feature_names: list[str]  # the name of each column of features

    def __init__(self, table: FrameTable, dwells: list[Dwell], valid_aps: list[ValidAP]):
        rows = []
        for number, ap in enumerate(valid_aps):
            frames = [table.heard[i] for i in np.flatnonzero(table.ap_numbers == number)]
            listen_s = sum(dwell.end - dwell.listen_start
                           for dwell in dwells if dwell.channel == ap.channel)
            cochannel = sum(other.channel == ap.channel for other in valid_aps) - 1
            rows.append(self.ap_descriptor(ap, frames, listen_s, cochannel, len(valid_aps)))
        self.features = np.array([list(row.values()) for row in rows], dtype=np.float32)
        self.feature_names = list(rows[0])

    @staticmethod
    def ap_descriptor(ap: ValidAP, frames: list[dict], listen_s: float,
                      cochannel: int, n_valid_aps: int) -> dict[str, float]:
        """DESCRIPTOR_FEATURES: one valid AP summarised over the whole window.
        """
        beacons = sorted((f for f in frames if f["beacon"]), key=lambda f: f["tx_start"])
        levels = [f["rssi"] for f in beacons]
        stations = {f["ta"] for f in frames if f["ta"] and f["ta"] != ap.mac}
        return {
            "beacon_rssi_mean": statistics.fmean(levels),
            "beacon_rssi_std": statistics.pstdev(levels),
            "beacon_rssi_min": min(levels),
            "beacon_rssi_max": max(levels),
            "beacon_rssi_last": levels[-1],
            "beacons_log1p": math.log1p(len(levels)),
            "bss_airtime_fraction": sum(f["dur"] for f in frames) / (listen_s * 1e6),
            "bss_transmitters_log1p": math.log1p(len(stations)),
            "cochannel_valid_aps": float(cochannel),
            "n_valid_aps": float(n_valid_aps),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/cache"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    scans = find_scans(args.runs_dir)
    print(f"{len(scans)} scans under {args.runs_dir}")
    print(f"projection: {describe(False)}")

    build = functools.partial(build_scan, out_dir=args.out)
    if args.workers <= 1:
        results = map(build, scans)
    else:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=args.workers)
        results = pool.map(build, scans, chunksize=8)
    index = []
    try:
        for number, row in enumerate(results, 1):
            if row is not None:
                index.append(row)
            if number % 200 == 0 or number == len(scans):
                print(f"[{number}/{len(scans)}] {len(index)} cached")
    finally:
        if args.workers > 1:
            pool.shutdown()

    if not index:
        sys.exit("no scans cached")
    frame_names, aggregate_names, descriptor_names = index[0]["feature_names"]
    np.savez(
        args.out / "_index.npz",
        scan_ids=np.array([row["scan_id"] for row in index]),
        topology_ids=np.array([row["topology_id"] for row in index]),
        n_valid_aps=np.array([row["n_valid_aps"] for row in index], np.int16),
        configured_n_aps=np.array([row["n_aps"] for row in index], np.int16),
        n_hotspots=np.array([row["n_hotspots"] for row in index], np.int16),
        frame_features=np.array(frame_names),
        cell_aggregates=np.array(aggregate_names),
        descriptor_features=np.array(descriptor_names),
        scan_description=np.array(describe(False)),
    )
    cells = np.array([row["n_cells"] for row in index])
    frames = np.array([row["n_frames"] for row in index])
    print(f"cached {len(index)} scans to {args.out}: "
          f"cells per scan {cells.min()}-{cells.max()}, "
          f"frames per scan median {int(np.median(frames))}, max {frames.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
