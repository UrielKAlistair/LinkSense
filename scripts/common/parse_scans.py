"""The scans of a sweep, and the APs each scan can choose between.

A scan is one client position in one topology. run_sweep.py simulates it once
per AP, each run joining a different AP, in run directories named
<topology>__c<scan>__ap<AP index>, and lists every topology with its scans in
_manifest.json. The runs of a scan are identical up to the join, so one
recording serves the whole scan and each run adds the label of its AP.

  find_scans()    groups the run directories into scans, and raises if a scan
                  lacks the run for one of its APs or a run directory is not in
                  the manifest.
  find_valid_aps()  given a scan's projected recording, returns its valid
                    APs: those with a decoded beacon, which the client could
                    join, each with its channel and the label of the run that
                    joined it.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

from scripts.common.projection import ProjectedScan


@dataclasses.dataclass(frozen=True)
class Scan:
    """One topology + client position: contains runs for each AP."""

    scan_id: str              # <topology>__c<scan>, e.g. t00007__c01
    topology_id: str
    runs: dict[int, Path]     # AP index -> run directory, in AP index order

    @property
    def recording(self) -> Path:
        """The first run, by AP index, that holds observation.csv."""
        for run in self.runs.values():
            if (run / "observation.csv").exists():
                return run
        raise FileNotFoundError(f"{self.scan_id}: no run holds observation.csv")


def find_scans(runs_dir: Path) -> list[Scan]:
    """Every scan in _manifest.json, in topology and scan order.

    Raises if a scan lacks the run for one of its APs, or a run directory is
    not in the manifest.
    """
    manifest = json.loads((runs_dir / "_manifest.json").read_text())
    unlisted = {p.name for p in runs_dir.iterdir() if p.is_dir()}
    scans = []
    for topology_id in sorted(manifest):
        entry = manifest[topology_id]
        for candidate in range(len(entry["candidates"])):
            scan_id = f"{topology_id}__c{candidate:02d}"
            runs = {ap: runs_dir / f"{scan_id}__ap{ap}"
                    for ap in range(entry["topology"]["nAPs"])}
            missing = [run.name for run in runs.values()
                       if not (run / "metadata.json").exists()]
            if missing:
                raise ValueError(f"{scan_id}: no metadata.json in {missing}")
            unlisted -= {run.name for run in runs.values()}
            scans.append(Scan(scan_id, topology_id, runs))
    if unlisted:
        raise ValueError(f"run directories not in the manifest: {sorted(unlisted)[:10]}")
    return scans


@dataclasses.dataclass(frozen=True)
class ValidAP:
    """An AP whose beacon the client decoded, so it could join it, and what
    joining it gave."""

    index: int                # its index in the deployment
    mac: str
    channel: int              # the channel its beacons were decoded on
    throughput_mbps: float
    associated: bool
    metadata: dict            # metadata.json of the run that joined it


def find_valid_aps(scan: Scan, projected: ProjectedScan) -> list[ValidAP] | None:
    """The APs with a decoded beacon, in AP index order.

    None when fewer than two APs qualify, since the scan then offers no choice.
    """
    channel_of = beacon_channels(projected.frames)
    valid_aps = []
    for index, run in scan.runs.items():
        metadata = json.loads((run / "metadata.json").read_text())
        if metadata["candidate"]["target_ap"] != index:
            raise ValueError(f"{run.name}: metadata.json says it joined AP "
                             f"{metadata['candidate']['target_ap']}")
        mac = next(ap["mac"] for ap in metadata["aps"] if ap["index"] == index).lower()
        if mac in channel_of:
            valid_aps.append(ValidAP(index, mac, channel_of[mac],
                                  float(metadata["candidate"]["throughput_mbps"]),
                                  bool(metadata["candidate"]["associated"]), metadata))
    if len(valid_aps) < 2:
        print(f"WARN: scan {scan.scan_id} heard only {len(valid_aps)} AP(s); skipping",
              file=sys.stderr)
        return None
    return valid_aps


def beacon_channels(frames: list[dict]) -> dict[str, int]:
    """The channel each BSSID's beacons were received on."""
    return {r["bssid"]: r["channel"] for r in frames if r["beacon"] and r["bssid"]}
