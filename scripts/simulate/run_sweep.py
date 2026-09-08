#!/usr/bin/env python3
"""Build a dataset of ns-3 simulation observations for AP selection.

This file runs "n" Topological configurations of between 2 and 8 APs, 
and random number of stations. Each topology is simulated once per AP,
so that the true throughput by joining each can be known.

The output is one directory per run, named at three levels:

    t00007            a topology: one deployment of APs and background stations
    t00007__c01       a candidate: that same client standing somewhere else
    t00007__c01__ap1  a run: that candidate joining AP 1

All of a candidate's runs share one recording of the observation period, 
so it is written to disk only once.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple


class Run(NamedTuple):
    """Everything one invocation of the simulator needs.

    topology and candidate are dicts of ns-3 command-line arguments, passed
    through unchanged; target_ap picks which AP the client joins, and rng_seed
    drives fading, contention and rate control. tag is the output directory.
    """
    tag: str
    topology: dict
    candidate: dict
    target_ap: int
    rng_seed: int


# ---------------------------------------------------------------------------
# The script has four stages:
#
# plan_runs turns the command line into a list of Runs. For each topology it
# draws a deployment (sample_topology) and a few places for the client to stand
# in it (sample_candidate), then emits one Run per candidate per AP. Nothing is
# simulated yet; this is only the plan.
#
# merge_manifest writes that plan into the output directory, so a later stage
# can tell what the directory is meant to contain. reconcile decides whether an
# entry already recorded there is the same topology asked for more candidates,
# which is allowed, or a different topology under the same name, which is not.
#
# already_done checks if any given Run has already been simulated and stored.
# This lets an interrupted sweep restart without redoing
#
# We then simulate the runs that are left, several at a time with multithreading.
#
# Three seeds divide the randomness, and every one of them is drawn here and
# resolved inside the simulator:
#
#     topologySeed   AP lattice, channel assignment, station positions and
#                    offered loads, associations, shadowing among them
#     candidateSeed  where the client stands, and its own shadowing links
#     rngSeed        fading, contention, rate control
#
# Holding topologySeed while varying candidateSeed is what lets one deployment
# be sampled from several places with its background left untouched.
# ---------------------------------------------------------------------------


def plan_runs(args) -> tuple[list[Run], dict]:
    """Every run this sweep should produce, and the manifest describing them."""
    runs = []
    manifest = {}
    for topology_index in range(args.n_topologies):
        topology_id = f"t{topology_index:05d}"
        # One generator per purpose, keyed by topology, so drawing more
        # candidates cannot shift the topology draws and topology 500 does not
        # depend on the 499 before it. String seeds, not hash(), which is
        # randomised per process.
        key = f"{args.sweep_seed}:{topology_id}"
        topology_rng = random.Random(f"topology:{key}")
        candidate_rng = random.Random(f"candidates:{key}")
        seed_rng = random.Random(f"seeds:{key}")

        topology = sample_topology(topology_rng)
        candidates = [sample_candidate(candidate_rng)
                      for _ in range(args.candidates_per_topology)]
        rng_seeds = [seed_rng.randrange(1, 2**31 - 1) for _ in candidates]
        manifest[topology_id] = {"topology": topology, "candidates": candidates,
                                 "seeds": rng_seeds}

        for index, (candidate, rng_seed) in enumerate(zip(candidates, rng_seeds)):
            for target_ap in range(topology["nAPs"]):
                runs.append(Run(f"{topology_id}__c{index:02d}__ap{target_ap}",
                                topology, candidate, target_ap, rng_seed))
    return runs, manifest


def sample_topology(rng: random.Random) -> dict:
    """One deployment: how many APs and stations, and which APs are hotspots."""
    n_aps = rng.choice([2, 3, 4, 6, 8])

    sta_counts = [rng.choice([3, 4, 5]) for _ in range(n_aps)]

    max_hotspots = math.ceil(n_aps / 3)
    if rng.random() < 0.25:
        hotspot_count = 0
    else:
        hotspot_count = rng.choices(range(1, max_hotspots + 1),
                                    weights=range(max_hotspots, 0, -1))[0]
    hotspot_aps = sorted(rng.sample(range(n_aps), hotspot_count))

    return {
        "nAPs": n_aps,
        "nSTAs": sum(sta_counts),
        "hotspotAPs": ",".join(map(str, hotspot_aps)) if hotspot_aps else "none",
        "topologySeed": rng.randrange(1, 2**31 - 1),
    }


def sample_candidate(rng: random.Random) -> dict:
    """Where the client stands, as a stratum the simulator resolves."""
    return {
        "candidateStratum": "boundary" if rng.random() < 0.70 else "ap_near",
        "candidateSeed": rng.randrange(1, 2**31 - 1),
    }


def merge_manifest(out_dir: Path, manifest: dict) -> None:
    """Add this sweep's topologies to the manifest, keeping earlier ones."""
    path = out_dir / "_manifest.json"
    stored = json.loads(path.read_text()) if path.exists() else {}
    merged = dict(stored)
    conflicts = []
    for topology_id, entry in manifest.items():
        agreed = (reconcile(stored[topology_id], entry)
                  if topology_id in stored else entry)
        if agreed is None:
            conflicts.append(topology_id)
        else:
            merged[topology_id] = agreed
    if conflicts:
        sys.exit(f"manifest already describes {len(conflicts)} of these topologies "
                 f"differently (first: {conflicts[:3]}). Use a new --out-dir, or "
                 "delete this one to start over.")
    kept = len(set(stored) - set(manifest))
    path.write_text(json.dumps(merged, indent=2))
    if kept:
        print(f"manifest: {len(manifest)} topologies written, {kept} earlier ones kept")


def reconcile(stored: dict, fresh: dict) -> dict | None:
    """Merge one topology's old and new manifest entries, or None if they clash.

    Topologies must be identical. Candidate and seed lists need only agree where
    they overlap; the longer list wins.
    """
    if stored["topology"] != fresh["topology"]:
        return None
    merged = {"topology": fresh["topology"]}
    for key in ("candidates", "seeds"):
        old, new = stored[key], fresh[key]
        longer, shorter = (old, new) if len(old) >= len(new) else (new, old)
        if longer[:len(shorter)] != shorter:
            return None
        merged[key] = longer
    return merged


def already_done(out_dir: Path, run: Run) -> bool:
    """True only when the files hold the exact run being asked for."""
    topology, candidate = run.topology, run.candidate
    run_dir = out_dir / run.tag
    metadata = run_dir / "metadata.json"
    # the simulator writes metadata last, so its absence means the run died
    if not metadata.exists() or metadata.stat().st_size == 0:
        return False
    try:
        stored = json.loads(metadata.read_text())
        params = stored["params"]
        expected_hotspots = ([] if topology["hotspotAPs"] == "none" else
                             [int(value) for value in topology["hotspotAPs"].split(",")])
        matches = (
            stored["rng_seed"] == run.rng_seed and
            stored["candidate"]["target_ap"] == run.target_ap and
            params["topology_seed"] == topology["topologySeed"] and
            params["n_aps"] == topology["nAPs"] and
            params["n_stas"] == topology["nSTAs"] and
            params["hotspot_aps"] == expected_hotspots and
            stored["candidate_seed"] == candidate["candidateSeed"] and
            stored["candidate_stratum"] == candidate["candidateStratum"]
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if not matches:
        return False
    if run.target_ap != 0:
        return True
    # ap0 carries the recording the candidate's runs share
    observation = run_dir / "observation.csv"
    channel_busy = run_dir / "chanbusy.csv"
    return (observation.exists() and observation.stat().st_size > 1000 and
            channel_busy.exists() and channel_busy.stat().st_size > 100)


def simulate(binary: str, out_dir: Path, run: Run,
             timeout: float) -> tuple[bool, str]:
    """Invoke the simulator once, returning success and any failure message."""
    settings = {
        **run.topology,
        **run.candidate,
        "outDir": out_dir,
        "runTag": run.tag,
        "targetAP": run.target_ap,
        "rngSeed": run.rng_seed,
        # a candidate's runs all hear the same thing, so only the first records it
        "captureObs": 1 if run.target_ap == 0 else 0,
    }
    args = [binary] + [f"--{key}={value}" for key, value in settings.items()]

    try:
        proc = subprocess.run(args, capture_output=True, text=True,
                              timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT {run.tag}"
    if proc.returncode != 0:
        return False, (f"FAILED {run.tag} rc={proc.returncode}\n"
                       f"{proc.stderr[-1200:]}")
    if not (out_dir / run.tag / "metadata.json").exists():
        return False, f"NO METADATA {run.tag}"
    return True, ""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-topologies", type=int, default=200,
                        help="each expands to candidates x nAPs runs")
    parser.add_argument("--candidates-per-topology", type=int, default=3,
                        help="candidate positions sampled within each topology")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sweep-seed", type=int, default=1234)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    if not args.binary.exists():
        sys.exit(f"binary not found: {args.binary}")
    for name in ("n_topologies", "candidates_per_topology", "workers"):
        if getattr(args, name) < 1:
            sys.exit(f"--{name.replace('_', '-')} must be at least 1")
    return args


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs, manifest = plan_runs(args)
    merge_manifest(args.out_dir, manifest)

    finished, pending = [], []
    for run in runs:
        target = finished if already_done(args.out_dir, run) else pending
        target.append(run)
    print(f"{args.n_topologies} topologies x {args.candidates_per_topology} "
          f"candidates -> {len(runs)} runs ({len(finished)} resumed, "
          f"{len(pending)} pending), {args.workers} at a time")
    print(f"writing to {args.out_dir}")

    start = time.time()
    n_ok = len(finished)
    n_fail = 0
    # threads, not processes: every run is a subprocess, so the GIL is idle
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(simulate, str(args.binary), args.out_dir, run,
                            args.timeout)
                for run in pending]
        for done, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            ok, msg = fut.result()
            n_ok += ok
            n_fail += not ok
            if not ok:
                print(msg, file=sys.stderr)
            if done % 50 == 0 or done == len(pending):
                elapsed = time.time() - start
                rate = done / elapsed if elapsed else 0
                eta = (len(pending) - done) / rate if rate else 0
                print(f"[{len(finished) + done}/{len(runs)}] ok={n_ok} "
                      f"fail={n_fail} elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m")

    print(f"done in {(time.time() - start) / 60:.1f}m: ok={n_ok} fail={n_fail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
