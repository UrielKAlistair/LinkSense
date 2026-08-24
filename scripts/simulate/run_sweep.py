#!/usr/bin/env python3
"""Generate the AP-selection dataset as matched sets ("choice sets").

The question this project asks is not "what throughput will this link give"
in isolation, but "given several APs I can hear, which should I join". So
the unit of data is a *group*: one scenario, held fixed, run once per
candidate AP.

Within a seed-realisation group everything is identical - AP and STA
placement, background load, the candidate's absolute position, and the RNG
seed - and only targetAP changes. The script can also repeat each topology
under several ns-3 seeds; those become separate choice sets with a shared
topology_id so training/evaluation splits can keep the physical topology
from leaking across train and test.

Two consequences worth stating explicitly:

  * The candidate observes every channel through dedicated scanner radios
    (one per channel, listening but never associating), so its view does
    not depend on which AP it later joins. Its association radio parks on
    an unused channel until join time for the same reason.

  * The candidate is placed in ABSOLUTE coordinates, not at an offset from
    its target. An offset would move the candidate whenever targetAP
    changed, which would compare different physical situations rather than
    different choices.

Randomness has two deliberately separate jobs. This script's RNG chooses the
physical scenario, including one topologySeed that fixes every background-STA
position. Each topology then receives several independent ns-3 rngSeed values
for fading, contention, and rate control; every target-AP variant within one
seed-realisation still shares that same ns-3 seed.
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


AP_SPACING_M = 30.0
DEFAULT_MAX_OUTPUT_GB = 10.0


def ap_positions(n_aps: int) -> list[tuple[float, float]]:
    """The fixed triangle/compact-grid rule implemented by the simulator."""
    if n_aps == 3:
        return [(0.0, 0.0),
                (AP_SPACING_M, 0.0),
                (AP_SPACING_M / 2.0, AP_SPACING_M * math.sqrt(3.0) / 2.0)]

    columns = n_aps // 2 if 4 <= n_aps <= 8 and n_aps % 2 == 0 else math.ceil(math.sqrt(n_aps))
    positions = []
    for index in range(n_aps):
        row, column = divmod(index, columns)
        count_in_row = min(columns, n_aps - row * columns)
        row_offset = (columns - count_in_row) * AP_SPACING_M / 2.0
        positions.append((row_offset + column * AP_SPACING_M,
                          row * AP_SPACING_M))
    return positions


def neighbouring_pairs(positions: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """Pairs one AP spacing apart: triangle edges or grid neighbours."""
    pairs = []
    for left in range(len(positions)):
        for right in range(left + 1, len(positions)):
            dx = positions[right][0] - positions[left][0]
            dy = positions[right][1] - positions[left][1]
            if math.hypot(dx, dy) <= AP_SPACING_M * 1.01:
                pairs.append((left, right))
    return pairs


def sample_scenario(rng: random.Random) -> dict:
    """One physical scenario, before any AP choice is made."""
    n_aps = rng.choice([2, 3, 4, 6, 8])
    positions = ap_positions(n_aps)
    n_stas = n_aps * rng.choice([2, 3, 4])

    # Most candidates sit near an edge shared by neighbouring AP cells. Signal
    # strength is then competitive enough that the load difference can matter.
    # The remaining cases sit near one AP and preserve straightforward
    # strongest-signal examples in the dataset.
    candidate_stratum = "boundary" if rng.random() < 0.70 else "ap_near"
    if candidate_stratum == "boundary":
        left, right = rng.choice(neighbouring_pairs(positions))
        ax, ay = positions[left]
        bx, by = positions[right]
        fraction = rng.uniform(0.35, 0.65)
        edge_x = ax + fraction * (bx - ax)
        edge_y = ay + fraction * (by - ay)
        length = math.hypot(bx - ax, by - ay)
        offset = rng.uniform(-0.20, 0.20) * AP_SPACING_M
        candidate_x = edge_x - offset * (by - ay) / length
        candidate_y = edge_y + offset * (bx - ax) / length
    else:
        anchor_x, anchor_y = positions[rng.randrange(n_aps)]
        distance = rng.uniform(4.0, 16.0)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        candidate_x = anchor_x + distance * math.cos(angle)
        candidate_y = anchor_y + distance * math.sin(angle)

    # A quarter of deployments are spatially uniform. The remainder contain
    # one or more crowded regions, with fewer hotspots more likely than the
    # maximum. The cap grows with the deployment but never exceeds three:
    # 1 for 2/3 APs, 2 for 4/6 APs, and 3 for 8 APs.
    max_hotspots = min(3, math.ceil(n_aps / 3))
    active_weights = list(range(max_hotspots, 0, -1))
    hotspot_count = rng.choices(
        range(max_hotspots + 1),
        weights=[sum(active_weights) / 3] + active_weights,
    )[0]
    hotspot_aps = sorted(rng.sample(range(n_aps), hotspot_count))

    return {
        "nAPs": n_aps,
        "nSTAs": n_stas,
        "candidateX": round(candidate_x, 3),
        "candidateY": round(candidate_y, 3),
        "candidateStratum": candidate_stratum,
        "hotspotAPs": ",".join(map(str, hotspot_aps)) if hotspot_aps else "none",
        "bgPerStaMbps": rng.choice([1.0, 2.5, 4.0, 6.0]),
        "topologySeed": rng.randrange(1, 2**31 - 1),
    }


def run_variant(binary: str, out_dir: Path, tag: str, scenario: dict, target_ap: int,
                seed: int, timeout: float) -> tuple[bool, str]:
    args = [binary, f"--outDir={out_dir}", f"--runTag={tag}",
            f"--targetAP={target_ap}", f"--rngSeed={seed}",
            # every variant would observe exactly the same thing, so only the
            # first one pays the cost of writing it down
            f"--captureObs={1 if target_ap == 0 else 0}"]
    args += [f"--{k}={v}" for k, v in scenario.items() if k != "candidateStratum"]

    try:
        proc = subprocess.run(args, capture_output=True, text=True,
                             timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT {tag}"
    if proc.returncode != 0:
        return False, f"FAILED {tag} rc={proc.returncode}\n{proc.stderr[-1200:]}"
    if not (out_dir / tag / "metadata.json").exists():
        return False, f"NO METADATA {tag}"
    return True, ""


def variant_complete(out_dir: Path, tag: str, scenario: dict,
                     target_ap: int, seed: int) -> bool:
    """True only when files contain the exact requested counterfactual run."""
    run_dir = out_dir / tag
    metadata = run_dir / "metadata.json"
    if not metadata.exists() or metadata.stat().st_size == 0:
        return False
    try:
        stored = json.loads(metadata.read_text())
        params = stored["params"]
        expected_hotspots = ([] if scenario["hotspotAPs"] == "none" else
                             [int(value) for value in scenario["hotspotAPs"].split(",")])
        matches = (
            stored["rng_seed"] == seed and
            stored["candidate"]["target_ap"] == target_ap and
            params["topology_seed"] == scenario["topologySeed"] and
            params["n_aps"] == scenario["nAPs"] and
            params["n_stas"] == scenario["nSTAs"] and
            math.isclose(params["bg_per_sta_mbps"], scenario["bgPerStaMbps"]) and
            params["hotspot_aps"] == expected_hotspots and
            math.isclose(stored["candidate_position"]["x"], scenario["candidateX"],
                         abs_tol=1e-6) and
            math.isclose(stored["candidate_position"]["y"], scenario["candidateY"],
                         abs_tol=1e-6)
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if not matches:
        return False
    if target_ap != 0:
        return True
    observation = run_dir / "observation.csv"
    channel_busy = run_dir / "chanbusy.csv"
    return (observation.exists() and observation.stat().st_size > 1000 and
            channel_busy.exists() and channel_busy.stat().st_size > 100)


def directory_size(path: Path) -> int:
    """Bytes currently stored below path, tolerating concurrently created files."""
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except FileNotFoundError:
            pass
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-groups", type=int, default=200,
                        help="topologies; each expands to seeds-per-topology x nAPs runs")
    parser.add_argument("--seeds-per-topology", type=int, default=5,
                        help="independent ns-3 seeds per physical topology")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sweep-seed", type=int, default=1234)
    parser.add_argument("--group-prefix", default="g",
                        help="run/topology ID prefix, useful when combining sweeps")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--max-output-gb", type=float, default=DEFAULT_MAX_OUTPUT_GB,
                        help="storage safety ceiling; scheduling stops at 95%% of this "
                             f"value (default {DEFAULT_MAX_OUTPUT_GB:g} GB)")
    args = parser.parse_args()

    if not args.binary.exists():
        sys.exit(f"binary not found: {args.binary}")
    if args.seeds_per_topology < 1:
        sys.exit("--seeds-per-topology must be at least 1")
    if args.n_groups < 1:
        sys.exit("--n-groups must be at least 1")
    if args.workers < 1:
        sys.exit("--workers must be at least 1")
    if args.max_output_gb <= 0:
        sys.exit("--max-output-gb must be positive")
    if (not args.group_prefix or "__" in args.group_prefix or
            not all(character.isalnum() or character in "-_"
                    for character in args.group_prefix)):
        sys.exit("--group-prefix must be nonempty, path-safe, and cannot contain '__'")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.sweep_seed)
    jobs: list[tuple[str, dict, int, int]] = []
    manifest = {}
    for g in range(args.n_groups):
        scenario = sample_scenario(rng)
        seeds = [rng.randrange(1, 2**31 - 1) for _ in range(args.seeds_per_topology)]
        group_id = f"{args.group_prefix}{g:05d}"
        manifest[group_id] = {"scenario": scenario, "seeds": seeds}
        for seed_idx, seed in enumerate(seeds):
            for target_ap in range(scenario["nAPs"]):
                jobs.append((f"{group_id}__s{seed_idx:02d}__ap{target_ap}",
                             scenario, target_ap, seed))

    (args.out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    completed_jobs = [job for job in jobs
                      if variant_complete(args.out_dir, *job)]
    pending_jobs = [job for job in jobs
                    if not variant_complete(args.out_dir, *job)]
    print(f"{args.n_groups} topologies x {args.seeds_per_topology} seeds "
          f"-> {len(jobs)} runs ({len(completed_jobs)} resumed, "
          f"{len(pending_jobs)} pending), {args.workers} at a time")
    print(f"writing to {args.out_dir}")

    max_bytes = int(args.max_output_gb * 1_000_000_000)
    # Leave generous headroom for the workers already in flight when a size
    # check crosses the threshold. The pilot's largest concurrent batch is
    # orders of magnitude smaller than this five-percent reserve.
    stop_bytes = int(max_bytes * 0.95)
    initial_bytes = directory_size(args.out_dir)
    if initial_bytes >= stop_bytes:
        sys.exit(f"output already uses {initial_bytes / 1e9:.2f} GB; "
                 f"refusing to approach the {args.max_output_gb:g} GB ceiling")

    start = time.time()
    n_ok = len(completed_jobs)
    n_fail = n_cancelled = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(run_variant, str(args.binary), args.out_dir, tag, sc, ap,
                           seed, args.timeout)
                for tag, sc, ap, seed in pending_jobs]
        stopped_for_size = False
        for done, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            if fut.cancelled():
                n_cancelled += 1
                continue
            ok, msg = fut.result()
            n_ok += ok
            n_fail += not ok
            if not ok:
                print(msg, file=sys.stderr)
            if (not stopped_for_size and done % 25 == 0 and
                    directory_size(args.out_dir) >= stop_bytes):
                stopped_for_size = True
                for queued in futs:
                    queued.cancel()
            processed = len(completed_jobs) + done
            if done % 50 == 0 or done == len(pending_jobs):
                el = time.time() - start
                rate = done / el if el else 0
                eta = (len(pending_jobs) - done) / rate if rate else 0
                print(f"[{processed}/{len(jobs)}] ok={n_ok} fail={n_fail} "
                      f"elapsed={el / 60:.1f}m eta={eta / 60:.1f}m")

    if stopped_for_size:
        print(f"storage guard stopped scheduling near {args.max_output_gb:g} GB; "
              f"cancelled={n_cancelled}")
    print(f"done in {(time.time() - start) / 60:.1f}m: ok={n_ok} fail={n_fail}")
    # A cap-triggered partial sweep is resumable, but it is not a completed
    # dataset and must not look successful to a calling pipeline.
    return 1 if n_fail or stopped_for_size else 0


if __name__ == "__main__":
    raise SystemExit(main())
