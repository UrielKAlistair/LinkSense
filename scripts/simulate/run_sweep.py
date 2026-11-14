#!/usr/bin/env python3
"""Build a dataset of ns-3 simulation observations for throughput prediction.

This file draws deployments over a wide range of regimes and simulates clients
arriving in them. The output is one directory per run, named at three levels:

    t00007              a topology: one deployment of rooms, APs and stations
    t00007__c01         a scan: one client standing somewhere in it
    t00007__c01__draw   the scan's observation, the client then joining an AP
                        the simulator draws among those it can hear
    t00007__c01__ap2    the same scan, the client joining AP 2 instead
    t00007__c01__f3     the same scan and AP as __draw, with different
                        background traffic after the join

PROCESS
  1. Draw each topology's regime and each scan's seeds.
  2. Simulate every scan's __draw run, which records the observation.
  3. In matched topologies, simulate the scan once per remaining AP.
  4. In replicated topologies, simulate the scan's drawn AP under further
     futures.

Runs whose metadata.json exists are skipped, so an interrupted sweep resumes.

Run:
  .venv/bin/python3 scripts/simulate/run_sweep.py \
      --binary ~/workspace/ns-3.45/build/scratch/ns3.45-my-wifi-test \
      --out-dir data/runs_v2 --n-topologies 1000 --workers 14
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

# The floor has two rows of rooms, so a topology of n_cols columns has
# 2 * n_cols rooms, and at most one AP per room.
ROOM_ROWS = 2
N_COLS = (2, 6)
MAX_APS = 8

# Every topology draws its regime from these ranges, the starred ones on a
# logarithmic scale.
MEAN_OCCUPANCY = (0.3, 4.0)      # * stations per room
OCCUPANCY_SHAPE = (0.3, 10.0)    # * small crowds a few rooms
LOAD_MEDIAN_MBPS = (0.3, 10.0)   # * per station, while in a session
DOWNLINK_FRACTION = (0.0, 1.0)
TCP_FRACTION = (0.0, 1.0)

SCANS_PER_TOPOLOGY = 5
# One topology in MATCHED_EVERY is simulated once per AP, and the next one
# under FUTURES_PER_REPLICATED_SCAN futures.
MATCHED_EVERY = 10
FUTURES_PER_REPLICATED_SCAN = 5


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scans = plan_scans(args)
    print(f"{args.n_topologies} topologies -> {len(scans)} scans, "
          f"{args.workers} at a time, writing to {args.out_dir}", flush=True)

    started, n_runs, n_failed = time.time(), 0, 0
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        finished = pool.map(lambda scan: simulate_scan(args, scan), scans)
        for done, (runs, failed) in enumerate(finished, start=1):
            n_runs, n_failed = n_runs + runs, n_failed + failed
            if done % 20 == 0 or done == len(scans):
                minutes = (time.time() - started) / 60
                print(f"[{done}/{len(scans)} scans] runs={n_runs} fail={n_failed} "
                      f"elapsed={minutes:.1f}m "
                      f"eta={minutes * (len(scans) - done) / done:.1f}m", flush=True)
    sys.exit(1 if n_failed else 0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-topologies", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--sweep-seed", type=int, default=1234)
    parser.add_argument("--session-shape", type=float, default=0.0,
                        help="Pareto shape of session and pause lengths; "
                             "0 keeps them exponential")
    parser.add_argument("--timeout", type=float, default=1800.0,
                        help="seconds one run may take")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 1. The plan
#
# A topology is a regime and a seed; a scan adds the client's seeds. Every
# draw comes from a generator keyed by the sweep seed and the topology's
# number, so growing the sweep leaves the earlier topologies as they were.
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Scan:
    """One client in one topology: the simulator's arguments, less the target."""

    tag: str
    arguments: dict
    is_matched: bool
    is_replicated: bool


def plan_scans(args) -> list[Scan]:
    scans = []
    for index in range(args.n_topologies):
        rng = random.Random(f"topology:{args.sweep_seed}:{index}")
        topology = sample_topology(rng)
        for candidate in range(SCANS_PER_TOPOLOGY):
            scans.append(Scan(
                tag=f"t{index:05d}__c{candidate:02d}",
                arguments={**topology,
                           "sessionShape": args.session_shape,
                           "candidateSeed": rng.randrange(1, 2**31 - 1),
                           "rngSeed": rng.randrange(1, 2**31 - 1)},
                is_matched=index % MATCHED_EVERY == 0,
                is_replicated=index % MATCHED_EVERY == 1))
    return scans


def sample_topology(rng: random.Random) -> dict:
    """One deployment's regime, as simulator arguments."""
    log_uniform = lambda low, high: math.exp(rng.uniform(math.log(low), math.log(high)))  # noqa: E731
    n_cols = rng.randint(*N_COLS)
    return {
        "topologySeed": rng.randrange(1, 2**31 - 1),
        "nCols": n_cols,
        "nAPs": rng.randint(2, min(MAX_APS, ROOM_ROWS * n_cols)),
        "meanOccupancy": round(log_uniform(*MEAN_OCCUPANCY), 4),
        "occupancyShape": round(log_uniform(*OCCUPANCY_SHAPE), 4),
        "loadMedianMbps": round(log_uniform(*LOAD_MEDIAN_MBPS), 4),
        "downlinkFraction": round(rng.uniform(*DOWNLINK_FRACTION), 4),
        "tcpFraction": round(rng.uniform(*TCP_FRACTION), 4),
    }


# ---------------------------------------------------------------------------
# 2-4. The runs of one scan
#
# The __draw run comes first because the simulator picks its target AP, and
# the other runs are defined by it: the matched runs cover the APs it did not
# pick, the replicated runs repeat the one it did.
# ---------------------------------------------------------------------------

def simulate_scan(args, scan: Scan) -> tuple[int, int]:
    """Simulates the scan's runs; returns how many it ran and how many failed."""
    drawn = simulate(args, f"{scan.tag}__draw", scan.arguments,
                     {"targetAP": -1, "captureObs": 1})
    if drawn is None:
        return 1, 1

    target = drawn["candidate"]["target_ap"]
    extras = []
    if scan.is_matched:
        extras = [(f"{scan.tag}__ap{ap}", {"targetAP": ap})
                  for ap in range(scan.arguments["nAPs"]) if ap != target]
    elif scan.is_replicated:
        # Under heavy-tailed sessions a future also redraws, given its age,
        # how long each period in force at the join goes on.
        is_redrawn = int(scan.arguments["sessionShape"] > 0)
        extras = [(f"{scan.tag}__f{future}", {"targetAP": target, "futureSeed": future,
                                              "redrawSessions": is_redrawn})
                  for future in range(2, FUTURES_PER_REPLICATED_SCAN + 1)]
    outcomes = [simulate(args, tag, scan.arguments, {**extra, "captureObs": 0})
                for tag, extra in extras]
    return 1 + len(outcomes), sum(outcome is None for outcome in outcomes)


def simulate(args, tag: str, arguments: dict, extra: dict) -> dict | None:
    """Runs the simulator once and returns its metadata, or None if it failed."""
    metadata_path = args.out_dir / tag / "metadata.json"
    if not metadata_path.exists():
        command = [str(args.binary), f"--outDir={args.out_dir}", f"--runTag={tag}",
                   *(f"--{name}={value}" for name, value in {**arguments, **extra}.items())]
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT {tag}", flush=True)
            return None
        if result.returncode != 0 or not metadata_path.exists():
            print(f"FAILED {tag}: {result.stderr.strip()[-300:]}", flush=True)
            return None
    return json.loads(metadata_path.read_text())


if __name__ == "__main__":
    main()
