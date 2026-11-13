"""The scans of a sweep, and the APs each scan could join.

A scan is one client position in one topology. run_sweep.py records it once, in
<scan>__draw, the client then joining an AP the simulator drew. Further runs
of the same scan share that recording up to the join: <scan>__ap<AP index>
joins another AP, and <scan>__f<n> joins the drawn AP again under different
background traffic.

  find_scans()    groups the run directories into scans.
  find_discovered_aps()  given a scan's projected recording, returns its
                         discovered APs: those with a decoded beacon, which the
                         client could join, each with its channel and, where a
                         run joined it, the throughput that run measured.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import sys
from pathlib import Path

from scripts.common.projection import ProjectedScan

RUN_NAME = re.compile(r"(?P<scan>(?P<topology>t\d+)__c\d+)__(?P<kind>draw|ap\d+|f\d+)")


@dataclasses.dataclass(frozen=True)
class Scan:
    """One topology + client position: its recording and its other runs."""

    scan_id: str              # <topology>__c<scan>, e.g. t00007__c01
    topology_id: str
    recording: Path           # the __draw run, which holds observation.csv
    other_runs: tuple[Path, ...]   # the __ap and __f runs


def find_scans(runs_dir: Path) -> list[Scan]:
    """Every scan whose __draw run finished, in topology and scan order."""
    finished = sorted(p for p in runs_dir.iterdir() if (p / "metadata.json").exists())
    others: dict[str, list[Path]] = {}
    for run in finished:
        name = RUN_NAME.fullmatch(run.name)
        if name["kind"] != "draw":
            others.setdefault(name["scan"], []).append(run)
    scans = []
    for run in finished:
        name = RUN_NAME.fullmatch(run.name)
        if name["kind"] == "draw":
            scans.append(Scan(name["scan"], name["topology"], run,
                              tuple(others.get(name["scan"], ()))))
    return scans


@dataclasses.dataclass(frozen=True)
class DiscoveredAP:
    """An AP whose beacon the client decoded, so it could join it, and what
    joining it gave in the runs that did."""

    index: int                # its index in the deployment
    mac: str
    channel: int              # the channel its beacons were decoded on
    throughput_mbps: float    # NaN when no run joined it
    replicate_mbps: tuple[float, ...]   # the same join under other futures


def find_discovered_aps(scan: Scan, projected: ProjectedScan) -> list[DiscoveredAP] | None:
    """The APs with a decoded beacon, in AP index order.

    The label of the AP the recording joined is the recording's own; an __ap
    run labels the AP it joined, and the __f runs are kept beside the drawn
    AP's label rather than averaged into it. None when no discovered AP has a
    label, since the scan then has nothing to learn from or be scored on.
    """
    recording = json.loads((scan.recording / "metadata.json").read_text())
    drawn = recording["candidate"]["target_ap"]
    label = {drawn: float(recording["candidate"]["throughput_mbps"])}
    replicates = []
    for run in scan.other_runs:
        candidate = json.loads((run / "metadata.json").read_text())["candidate"]
        if RUN_NAME.fullmatch(run.name)["kind"].startswith("f"):
            replicates.append(float(candidate["throughput_mbps"]))
        else:
            label[candidate["target_ap"]] = float(candidate["throughput_mbps"])

    channel_of = beacon_channels(projected.frames)
    discovered_aps = [
        DiscoveredAP(ap["index"], ap["mac"].lower(), channel_of[ap["mac"].lower()],
                     label.get(ap["index"], math.nan),
                     tuple(replicates) if ap["index"] == drawn else ())
        for ap in recording["aps"] if ap["mac"].lower() in channel_of]
    if all(math.isnan(ap.throughput_mbps) for ap in discovered_aps):
        print(f"WARN: scan {scan.scan_id} did not discover the AP it joined; skipping",
              file=sys.stderr)
        return None
    return discovered_aps


def beacon_channels(frames: list[dict]) -> dict[str, int]:
    """The channel each BSSID's beacons were received on."""
    return {r["bssid"]: r["channel"] for r in frames if r["beacon"] and r["bssid"]}
