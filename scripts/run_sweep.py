#!/usr/bin/env python3
"""Orchestrate many ns-3 my-wifi-test runs, sweeping scenario parameters.

Each invocation of the compiled ns-3 binary produces one run directory
(metadata.json + candidate-side pcap) under --out-dir. This script decides
*which* scenarios to run (topology, targetAP, candidate placement, channel
plan) and fires off the binary once per scenario, in parallel.

Two separate sources of randomness are deliberately kept apart:
  - This script's own `random.Random(args.sweep_seed)` picks which *scenario*
    to run (topology knobs, targetAP, candidate placement) - reproducible
    given the same --sweep-seed and --n-runs.
  - Each ns-3 process draws its own internal RNG seed (unless --fixed-rng-seed
    is passed) and records the seed it used in that run's metadata.json - so
    packet-level randomness (backoff, fading, ...) varies run to run even for
    repeated scenario configs, without this script needing to manage it.
"""

import argparse
import concurrent.futures
import random
import subprocess
import sys
import time
from pathlib import Path


def sample_scenario(rng: random.Random) -> dict:
    n_aps = rng.choice([2, 2, 3, 3, 4])  # weight toward 2-3 APs
    n_stas = rng.choice([2, 4, 6, 8, 12, 16])
    ap_spacing = rng.uniform(20.0, 60.0)
    target_ap = rng.randrange(n_aps)
    # candidate distance spans "very close" to "past the next AP over", so
    # the dataset covers both easy and marginal/poor association choices
    candidate_distance = rng.uniform(2.0, 1.5 * ap_spacing)
    candidate_angle_deg = rng.uniform(0.0, 360.0)
    n_channels = rng.choice([1, 1, 2, 3])  # weight toward co-channel (harder/more common case)
    jitter_std = rng.uniform(0.5, 4.0)
    packet_size = rng.choice([512, 1250, 1500])

    return {
        "nAPs": n_aps,
        "nSTAs": n_stas,
        "targetAP": target_ap,
        "apSpacing": round(ap_spacing, 3),
        "candidateDistance": round(candidate_distance, 3),
        "candidateAngleDeg": round(candidate_angle_deg, 3),
        "nChannels": n_channels,
        "jitterStd": round(jitter_std, 3),
        "packetSize": packet_size,
    }


def run_one(binary: str, out_dir: Path, scenario: dict, sim_stop_time: float,
            candidate_start_time: float, fixed_rng_seed: int, timeout: float) -> tuple[bool, str]:
    args = [binary,
            f"--outDir={out_dir}",
            f"--simStopTime={sim_stop_time}",
            f"--candidateStartTime={candidate_start_time}"]
    for key, value in scenario.items():
        args.append(f"--{key}={value}")
    if fixed_rng_seed:
        args.append(f"--rngSeed={fixed_rng_seed}")

    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT: {' '.join(args)}"

    if result.returncode != 0:
        return False, f"FAILED (rc={result.returncode}): {' '.join(args)}\n{result.stderr[-2000:]}"
    return True, ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True,
                        help="path to the compiled ns3.45-my-wifi-test binary")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="root directory to write per-run subdirectories under")
    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel ns-3 processes; keep modest, each run is CPU-heavy")
    parser.add_argument("--sweep-seed", type=int, default=1234,
                        help="seed for THIS script's scenario sampling (not ns-3's internal RNG)")
    parser.add_argument("--sim-stop-time", type=float, default=12.0)
    parser.add_argument("--candidate-start-time", type=float, default=6.0)
    parser.add_argument("--fixed-rng-seed", type=int, default=0,
                        help="if nonzero, pass this as every run's --rngSeed (mainly for debugging determinism); default 0 lets each run draw its own")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="per-run subprocess timeout in seconds")
    args = parser.parse_args()

    if not args.binary.exists():
        sys.exit(f"binary not found: {args.binary}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scenario_rng = random.Random(args.sweep_seed)
    scenarios = [sample_scenario(scenario_rng) for _ in range(args.n_runs)]

    print(f"Launching {len(scenarios)} runs, {args.workers} at a time, into {args.out_dir}")
    start = time.time()
    n_ok = 0
    n_fail = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(run_one, str(args.binary), args.out_dir, scenario,
                       args.sim_stop_time, args.candidate_start_time,
                       args.fixed_rng_seed, args.timeout)
            for scenario in scenarios
        ]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            ok, msg = fut.result()
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                print(msg, file=sys.stderr)
            if i % 10 == 0 or i == len(scenarios):
                elapsed = time.time() - start
                print(f"[{i}/{len(scenarios)}] ok={n_ok} fail={n_fail} elapsed={elapsed:.1f}s")

    print(f"Done. ok={n_ok} fail={n_fail} total_time={time.time() - start:.1f}s")
    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
