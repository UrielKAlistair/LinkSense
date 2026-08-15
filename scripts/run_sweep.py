#!/usr/bin/env python3
"""Generate the AP-selection dataset as matched sets ("choice sets").

The question this project asks is not "what throughput will this link give"
in isolation, but "given several APs I can hear, which should I join". So
the unit of data is a *group*: one scenario, held fixed, run once per
candidate AP.

Within a group everything is identical - AP and STA placement, background
load, the candidate's absolute position, and the RNG seed - and only
targetAP changes. Because the candidate is passive before it joins, and
all APs share one channel, the pre-association capture window is
bit-identical across the variants of a group (verified: the frame-level
digest of the window matches across variants). One observation, therefore,
with one label per AP option - exactly the structure of a ranking problem.

Two consequences worth stating explicitly:

  * The candidate observes every channel through dedicated scanner radios
    (one per channel, listening but never associating), so its view does
    not depend on which AP it later joins. Its association radio parks on
    an unused channel until join time for the same reason.

  * The candidate is placed in ABSOLUTE coordinates, not at an offset from
    its target. An offset would move the candidate whenever targetAP
    changed, which would compare different physical situations rather than
    different choices.

Randomness comes from two separate, deliberately independent sources: this
script's own RNG chooses scenarios (reproducible via --sweep-seed), while
each group draws one ns-3 seed shared by all its variants.
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


# A single 802.11n 20 MHz channel carries roughly 55 Mbps of UDP goodput
# here (see scripts/validate_sim.py, V1 at short range). Offered background
# load is sampled as a fraction of that rather than per-STA, because
# per-STA sampling multiplied by the STA count can oversubscribe the medium
# many times over - at which point every AP is equally hopeless, every label
# collapses to zero, and the choice carries no information.
CHANNEL_CAPACITY_MBPS = 55.0


def sample_scenario(rng: random.Random) -> dict:
    """One physical scenario, before any AP choice is made."""
    n_aps = rng.choice([2, 2, 3, 3, 3, 4])
    ap_spacing = rng.uniform(15.0, 45.0)
    n_stas = rng.choice([2, 4, 6, 8, 10, 14, 18])

    # Total offered load and client count are sampled independently: load
    # sets how full the medium is, while the number of clients sets how
    # much contention overhead produces that load. Conflating them (as
    # per-STA sampling does) makes the two effects impossible to separate.
    total_offered = rng.uniform(0.05, 1.05) * CHANNEL_CAPACITY_MBPS

    # Where the candidate stands decides whether the choice is interesting.
    # Sitting next to one AP makes that AP the answer almost regardless of
    # load, so a purely AP-anchored sample yields a dataset a
    # strongest-signal rule already solves (measured: 91% top-1). Half the
    # scenarios are therefore placed between two adjacent APs, where signal
    # is nearly tied and load is what actually decides.
    if n_aps >= 2 and rng.random() < 0.55:
        left = rng.randrange(n_aps - 1)
        frac = rng.uniform(0.3, 0.7)  # somewhere in the contested middle
        candidate_x = (left + frac) * ap_spacing
        candidate_y = rng.uniform(-0.35, 0.35) * ap_spacing
    else:
        anchor = rng.randrange(n_aps)
        dist = rng.uniform(2.0, 30.0)
        ang = rng.uniform(0.0, 2.0 * math.pi)
        candidate_x = anchor * ap_spacing + dist * math.cos(ang)
        candidate_y = dist * math.sin(ang)

    # Concentrate the background load on one AP most of the time, so the
    # options differ in how busy they are and not only in how far away.
    if rng.random() < 0.7:
        cluster_ap = rng.randrange(n_aps)
        cluster_frac = rng.uniform(0.5, 0.95)
        cluster_radius = rng.uniform(5.0, 15.0)
    else:
        cluster_ap, cluster_frac, cluster_radius = -1, 0.0, 10.0

    return {
        "staClusterAp": cluster_ap,
        "staClusterFrac": round(cluster_frac, 3),
        "staClusterRadius": round(cluster_radius, 2),
        "nAPs": n_aps,
        "nSTAs": n_stas,
        "apSpacing": round(ap_spacing, 3),
        "candidateX": round(candidate_x, 3),
        "candidateY": round(candidate_y, 3),
        "candidateAbsolute": 1,
        "jitterStd": round(rng.uniform(0.5, 5.0), 3),
        "bgPerStaMbps": round(total_offered / n_stas, 4),
        "packetSize": rng.choice([512, 1000, 1250, 1500]),
        # Channel plan. With every AP co-channel the medium is shared no
        # matter which AP is chosen, so load cannot differentiate the
        # options and the decision collapses to "pick the strongest signal"
        # (measured: a strongest-RSSI rule was optimal in 97% of co-channel
        # groups). Separating APs onto non-overlapping channels is what
        # makes load an AP-specific property, and is what real deployments
        # do. Co-channel cases are kept as the hard, dense-deployment
        # minority rather than the default.
        "nChannels": rng.choice([1, 2, 2, 3, 3, min(n_aps, 4)]),
    }


def run_variant(binary: str, out_dir: Path, tag: str, scenario: dict, target_ap: int,
                seed: int, sim_stop: float, cand_start: float,
                timeout: float) -> tuple[bool, str]:
    args = [binary, f"--outDir={out_dir}", f"--runTag={tag}",
            f"--targetAP={target_ap}", f"--rngSeed={seed}",
            f"--simStopTime={sim_stop}", f"--candidateStartTime={cand_start}",
            # every variant would observe exactly the same thing, so only the
            # first one pays the cost of writing it down
            f"--captureObs={1 if target_ap == 0 else 0}"]
    args += [f"--{k}={v}" for k, v in scenario.items()]

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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-groups", type=int, default=400,
                        help="scenarios; each expands to nAPs runs")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sweep-seed", type=int, default=1234)
    parser.add_argument("--sim-stop-time", type=float, default=17.0)
    parser.add_argument("--candidate-start-time", type=float, default=5.0,
                        help="end of the pre-association feature window")
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    if not args.binary.exists():
        sys.exit(f"binary not found: {args.binary}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.sweep_seed)
    jobs: list[tuple[str, dict, int, int]] = []
    manifest = {}
    for g in range(args.n_groups):
        scenario = sample_scenario(rng)
        seed = rng.randrange(1, 2**31 - 1)
        group_id = f"g{g:05d}"
        manifest[group_id] = {"scenario": scenario, "seed": seed}
        for target_ap in range(scenario["nAPs"]):
            jobs.append((f"{group_id}__ap{target_ap}", scenario, target_ap, seed))

    (args.out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{args.n_groups} groups -> {len(jobs)} runs, {args.workers} at a time")
    print(f"writing to {args.out_dir}")

    start = time.time()
    n_ok = n_fail = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(run_variant, str(args.binary), args.out_dir, tag, sc, ap,
                           seed, args.sim_stop_time, args.candidate_start_time,
                           args.timeout)
                for tag, sc, ap, seed in jobs]
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            ok, msg = fut.result()
            n_ok += ok
            n_fail += not ok
            if not ok:
                print(msg, file=sys.stderr)
            if i % 50 == 0 or i == len(jobs):
                el = time.time() - start
                rate = i / el if el else 0
                eta = (len(jobs) - i) / rate if rate else 0
                print(f"[{i}/{len(jobs)}] ok={n_ok} fail={n_fail} "
                      f"elapsed={el / 60:.1f}m eta={eta / 60:.1f}m")

    print(f"done in {(time.time() - start) / 60:.1f}m: ok={n_ok} fail={n_fail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
