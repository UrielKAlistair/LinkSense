#!/usr/bin/env python3
"""Controlled experiments that check the ns-3 simulator behaves like real WiFi.

Every number this project trains on comes out of one simulator, and a simulator
that quietly returns plausible-looking throughput is indistinguishable, from
anywhere downstream of it, from one that models radio physics. These experiments
are how the two are told apart. Each varies one property of a scenario, holds
the rest still, and asserts the shape of the answer rather than its value --
there is no real deployment to compare against, so what can be checked is that
throughput falls when the client walks away, falls when its neighbours get
busier, and does not move when nothing that should matter has changed.

One run of the simulator is one client joining one AP in one deployment, and
three seeds divide its randomness. topologySeed fixes the world: where the APs
and background stations stand, which channel each AP picks, how loud each
station is. candidateSeed fixes the client: where it stands, and how much each
of its own radio links is attenuated. rngSeed drives fading, contention and rate
control. Most of the checks below work by holding two of the three still and
moving the third.

  V1 distance     One AP, one idle neighbour, the client walked from 1 m out to
                  60 m. Throughput should sit near the link's ceiling up close
                  and near zero far away, with no step back up in between. Runs
                  with shadowing switched off so that distance is the only thing
                  changing along the curve.
  V2 load         The client parked at 10 m while four neighbours are told to
                  send progressively more. Throughput should fall as they do.
  V3 interaction  A shorter distance walk crossed with three background loads,
                  four neighbours throughout. Every row should fall as the
                  neighbours get louder, and the idle column should still fall
                  with distance.
  V4 layout       AP positions follow from the AP count alone; channels are
                  drawn per AP from topologySeed. Checks the lattice is the
                  expected triangular one, that channels stay inside the four
                  allowed, that they hold still when only rngSeed moves, and
                  that they change when topologySeed does.
  V5 repeat       The same run twice produces the same label.
  V6 radio seed   One scenario at eight rngSeeds. The label should vary, because
                  fading and contention are real, but not so much that it is
                  mostly noise -- and the background stations must not move.
  V7 matched set  Every AP the client could have joined shares a single
                  recording of the period before it joined. Checked at the bytes.
  V8 shadowing    Each pair of nodes gets one fixed log-normal attenuation, and
                  the client's share of those comes from candidateSeed. Changing
                  that seed alone should move the label and nothing else; with
                  the shadowing sigma set to zero it should move nothing at all.

Run:  .venv/bin/python3 scripts/simulate/validate_sim.py --binary <path>

      --only V1,V8   run just those checks (each is worth minutes of CPU)
      --out-dir DIR  keep the simulator output on disk under DIR
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, NamedTuple

# Added to every simulator invocation that does not override them.
# candidateX = 0 puts the client on the y axis directly above AP 0, which the
# simulator always places at the origin, so candidateY alone is its distance
# from that AP -- the checks that call candidateY "distance" depend on this.
# One fixed topologySeed means every check that does not deliberately change
# the world sees the same one, so a failure is a change in the simulator and
# not a change of scenario.
SHARED_ARGS = {
    "candidateX": 0.0,
    "topologySeed": 2025,
}

# rngSeed values used wherever a check needs repeats. V1, V2 and V3 average the
# first --repeats of these; V6 uses all eight to measure a spread. Any distinct
# positive values would serve; they are listed rather than computed so that a
# failing run can be reproduced by copying the number out of the tag.
RADIO_SEEDS = (11, 23, 37, 41, 53, 67, 71, 83)

# candidateSeed values for V8. With no candidateStratum set the simulator uses
# candidateX/candidateY as passed, so these change the client's shadowing links
# and nothing else.
CANDIDATE_SEEDS = (101, 202, 303, 404, 505, 606)

# Passing bgPerStaMbps=0 tells the simulator to draw each station's offered load
# from its log-normal. A near-zero fixed rate is how a check asks for a
# background that is present but effectively idle.
IDLE_BACKGROUND_MBPS = 0.05

# Every point is a mean over noisy runs, so the monotonicity checks let a value
# rise this far above its predecessor before failing. Widening these hides real
# regressions; narrowing them makes the checks fail on fading alone. Contended
# scenarios are noisier than the single-station one and get the wider allowance.
QUIET_TOLERANCE_MBPS = 2.0
CONTENDED_TOLERANCE_MBPS = 3.0

# V1's two endpoints. A client a metre from an idle AP should reach most of the
# link's capacity; one 60 m away through path loss with no shadowing to help it
# should reach almost none.
NEAR_FLOOR_MBPS = 30.0
FAR_CEILING_MBPS = 5.0

# V4. APs sit on a triangular lattice of 20 m spacing: neighbours in a row are
# 20 m apart along x, consecutive rows are one equilateral triangle's height
# apart along y, and odd rows are shifted half a spacing right. The expected
# positions are written out in metres below; only the row pitch needs computing.
ROW_PITCH_M = 20.0 * math.sqrt(3.0) / 2.0
POSITION_TOLERANCE_M = 1e-3
ALLOWED_CHANNELS = {36, 40, 44, 48}
LAYOUT_TOPOLOGY_SEED = 7
# A second world for V4 to compare channel plans against.
OTHER_TOPOLOGY_SEED = 8

# V6. The radio seed must move the label, or fading and contention are not being
# simulated at all -- but if it swings throughput by more than this fraction of
# its own mean, the label carries more radio noise than scenario.
MIN_SEED_SPREAD_MBPS = 0.01
MAX_SEED_SPREAD_FRACTION = 0.5

# V7. A run that recorded no frames or no CCA buckets would make every variant's
# file identical for the wrong reason, so both are required to have real content.
MIN_OBSERVATION_BYTES = 1000
MIN_CHANNEL_BUSY_BYTES = 100

# V8. Shadowing has to move the label clear of ordinary run-to-run jitter.
MIN_SHADOWING_SPREAD_MBPS = 1.0

# One '#' in a printed bar stands for this much throughput.
MBPS_PER_BAR_CHAR = 1.5


class Verdict(NamedTuple):
    """One check's result: what it claims, whether it held, and the evidence."""
    label: str
    passed: bool
    detail: str


class Sim(NamedTuple):
    """The binary, the directory its runs go in, and how many to run at once.

    radio_seeds is the set V1, V2 and V3 average each of their points over.
    Checks that need only one radio seed take it from RADIO_SEEDS directly, so
    that --repeats cannot change what they test.
    """
    binary: str
    out_dir: Path
    workers: int
    radio_seeds: tuple[int, ...]

    def run(self, tag: str, **args) -> dict:
        """Simulate once into out_dir/tag and return the run's metadata."""
        argv = [self.binary, f"--outDir={self.out_dir}", f"--runTag={tag}"]
        for name, value in {**SHARED_ARGS, **args}.items():
            argv.append(f"--{name}={value}")
        proc = subprocess.run(argv, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"run failed: {' '.join(argv)}\n{proc.stderr[-1500:]}")
        return json.loads((self.out_dir / tag / "metadata.json").read_text())

    def run_all(self, jobs: list[tuple[str, dict]]) -> dict[str, dict]:
        """Simulate every (tag, args) job, several at a time, keyed by tag."""
        results: dict[str, dict] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self.run, tag, **args): tag for tag, args in jobs}
            for future in concurrent.futures.as_completed(futures):
                results[futures[future]] = future.result()
        return results


def throughput(meta: dict) -> float:
    """The label: what the client actually achieved on the AP it joined."""
    return meta["candidate"]["throughput_mbps"]


def is_monotone_decreasing(values: list[float], tolerance_mbps: float) -> bool:
    """True when no value exceeds the one before it by more than the tolerance."""
    return all(later <= earlier + tolerance_mbps
               for earlier, later in zip(values, values[1:]))


def bar(mbps: float) -> str:
    """A row of '#' proportional to throughput, for eyeballing a printed curve."""
    return "#" * int(mbps / MBPS_PER_BAR_CHAR)


def check_distance_decay(sim: Sim) -> Verdict:
    """V1: throughput against distance from a lone AP with an idle neighbour."""
    distances_m = [1, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60]
    # Shadowing is one fixed draw per node pair, and the client and the AP do
    # not move relative to each other between the points of this curve, so a
    # favourable draw would lift every point by the same number of dB and put
    # the far end above FAR_CEILING_MBPS. Zero leaves path loss alone on the
    # x axis. Removing this argument reintroduces that bias.
    jobs = [(f"v1_d{distance}_s{seed}",
             dict(nAPs=1, nSTAs=1, candidateY=distance,
                  bgPerStaMbps=IDLE_BACKGROUND_MBPS, linkShadowingDb=0.0,
                  rngSeed=seed))
            for distance in distances_m for seed in sim.radio_seeds]
    runs = sim.run_all(jobs)

    curve = [statistics.fmean(throughput(runs[f"v1_d{distance}_s{seed}"])
                              for seed in sim.radio_seeds)
             for distance in distances_m]
    print("V1 throughput vs distance (one AP, idle background, no shadowing)")
    for distance, mbps in zip(distances_m, curve):
        print(f"    {distance:>4} m  {mbps:>7.2f} Mbps  {bar(mbps)}")

    passed = (is_monotone_decreasing(curve, QUIET_TOLERANCE_MBPS)
              and curve[0] > NEAR_FLOOR_MBPS
              and curve[-1] < FAR_CEILING_MBPS)
    return Verdict("V1 throughput decays smoothly with distance", passed,
                   f"{curve[0]:.1f} -> {curve[-1]:.1f} Mbps")


def check_load_decay(sim: Sim) -> Verdict:
    """V2: throughput against how much the client's neighbours are sending."""
    loads_mbps = [IDLE_BACKGROUND_MBPS, 2, 4, 6, 8, 10, 12]
    jobs = [(f"v2_l{load}_s{seed}",
             dict(nAPs=1, nSTAs=4, candidateY=10, bgPerStaMbps=load, rngSeed=seed))
            for load in loads_mbps for seed in sim.radio_seeds]
    runs = sim.run_all(jobs)

    curve = [statistics.fmean(throughput(runs[f"v2_l{load}_s{seed}"])
                              for seed in sim.radio_seeds)
             for load in loads_mbps]
    print("\nV2 throughput vs background load (4 stations, client at 10 m)")
    for load, mbps in zip(loads_mbps, curve):
        print(f"    {load:>5} Mbps/STA ({load * 4:>5.1f} total)  "
              f"{mbps:>7.2f} Mbps  {bar(mbps)}")

    passed = (is_monotone_decreasing(curve, CONTENDED_TOLERANCE_MBPS)
              and curve[0] > curve[-1])
    return Verdict("V2 background load reduces client throughput", passed,
                   f"{curve[0]:.1f} -> {curve[-1]:.1f} Mbps")


def check_distance_load_grid(sim: Sim) -> Verdict:
    """V3: the distance curve redrawn at three background loads."""
    distances_m = [5, 15, 25, 35]
    loads_mbps = [IDLE_BACKGROUND_MBPS, 4, 8]
    jobs = [(f"v3_d{distance}_l{load}_s{seed}",
             dict(nAPs=1, nSTAs=4, candidateY=distance, bgPerStaMbps=load,
                  rngSeed=seed))
            for distance in distances_m for load in loads_mbps
            for seed in sim.radio_seeds]
    runs = sim.run_all(jobs)

    print("\nV3 distance x background load")
    print(f"    {'dist':>6}" + "".join(f"{f'{load}Mbps':>11}" for load in loads_mbps))
    rows = {}
    for distance in distances_m:
        rows[distance] = [
            statistics.fmean(throughput(runs[f"v3_d{distance}_l{load}_s{seed}"])
                             for seed in sim.radio_seeds)
            for load in loads_mbps]
        print(f"    {distance:>6}" + "".join(f"{v:>11.2f}" for v in rows[distance]))

    falls_with_load = all(is_monotone_decreasing(rows[d], CONTENDED_TOLERANCE_MBPS)
                          for d in distances_m)
    idle_column = [rows[d][0] for d in distances_m]
    falls_with_distance = is_monotone_decreasing(idle_column, CONTENDED_TOLERANCE_MBPS)
    return Verdict("V3 distance and load each degrade throughput",
                   falls_with_load and falls_with_distance,
                   f"{'rows ok' if falls_with_load else 'ROW FAIL'}, "
                   f"{'idle column ok' if falls_with_distance else 'COLUMN FAIL'}")


def check_layout_and_channels(sim: Sim) -> Verdict:
    """V4: AP positions follow the AP count; channels follow topologySeed."""
    expected_positions_m = {
        2: [(0.0, 0.0), (20.0, 0.0)],
        3: [(0.0, 0.0), (20.0, 0.0), (10.0, ROW_PITCH_M)],
        4: [(0.0, 0.0), (20.0, 0.0), (10.0, ROW_PITCH_M), (30.0, ROW_PITCH_M)],
        6: [(0.0, 0.0), (20.0, 0.0), (40.0, 0.0),
            (10.0, ROW_PITCH_M), (30.0, ROW_PITCH_M), (50.0, ROW_PITCH_M)],
        8: [(0.0, 0.0), (20.0, 0.0), (40.0, 0.0), (60.0, 0.0),
            (10.0, ROW_PITCH_M), (30.0, ROW_PITCH_M), (50.0, ROW_PITCH_M),
            (70.0, ROW_PITCH_M)],
    }
    first_seed, second_seed = RADIO_SEEDS[0], RADIO_SEEDS[1]

    print("\nV4 AP layout and channel plan")
    all_ok = True
    plans = {}
    for n_aps, expected in expected_positions_m.items():
        base = sim.run(f"v4_n{n_aps}", nAPs=n_aps, nSTAs=n_aps, candidateY=10,
                       topologySeed=LAYOUT_TOPOLOGY_SEED, rngSeed=first_seed)
        other_radio = sim.run(f"v4_n{n_aps}_radio", nAPs=n_aps, nSTAs=n_aps,
                              candidateY=10, topologySeed=LAYOUT_TOPOLOGY_SEED,
                              rngSeed=second_seed)

        positions = [(ap["position"]["x"], ap["position"]["y"]) for ap in base["aps"]]
        plans[n_aps] = [ap["channel"] for ap in base["aps"]]
        lattice_ok = all(abs(x - ex) < POSITION_TOLERANCE_M
                         and abs(y - ey) < POSITION_TOLERANCE_M
                         for (x, y), (ex, ey) in zip(positions, expected))
        channels_allowed = set(plans[n_aps]) <= ALLOWED_CHANNELS
        ignores_radio_seed = plans[n_aps] == [ap["channel"] for ap in other_radio["aps"]]

        ok = lattice_ok and channels_allowed and ignores_radio_seed
        all_ok = all_ok and ok
        print(f"    nAPs={n_aps}: channels={plans[n_aps]} "
              f"{'ok' if ok else 'MISMATCH'}")

    # The plan must also move when the world does, or every deployment would
    # reuse channels the same way. Checked at the widest layout only: with four
    # channels two independent two-AP plans coincide once in sixteen, so asking
    # this of the small layouts would make the check flaky.
    widest = max(expected_positions_m)
    other_world = sim.run(f"v4_n{widest}_world", nAPs=widest, nSTAs=widest,
                          candidateY=10, topologySeed=OTHER_TOPOLOGY_SEED,
                          rngSeed=first_seed)
    other_plan = [ap["channel"] for ap in other_world["aps"]]
    follows_topology_seed = plans[widest] != other_plan
    all_ok = all_ok and follows_topology_seed
    print(f"    nAPs={widest} at topologySeed {OTHER_TOPOLOGY_SEED}: {other_plan} "
          f"{'differs' if follows_topology_seed else 'IDENTICAL'}")

    return Verdict("V4 fixed lattice, channels drawn from topologySeed alone",
                   all_ok, "2/3/4/6/8 AP layouts")


def check_repeatability(sim: Sim) -> Verdict:
    """V5: the same run twice must produce the same label."""
    scenario = dict(nAPs=3, nSTAs=9, candidateY=12, rngSeed=RADIO_SEEDS[0])
    first = throughput(sim.run("v5_first", **scenario))
    second = throughput(sim.run("v5_second", **scenario))
    identical = first == second
    print(f"\nV5 repeatability: {first:.6f} vs {second:.6f} -> "
          f"{'identical' if identical else 'DIFFERENT'}")
    return Verdict("V5 the same seeds reproduce the same label", identical,
                   f"{first:.4f} Mbps")


def check_radio_seed_spread(sim: Sim) -> Verdict:
    """V6: rngSeed should move the label without dominating it, and move nothing else."""
    jobs = [(f"v6_s{seed}",
             dict(nAPs=3, nSTAs=9, candidateY=18, hotspotAPs="1", bgPerStaMbps=6,
                  rngSeed=seed))
            for seed in RADIO_SEEDS]
    runs = sim.run_all(jobs)

    labels = [throughput(runs[f"v6_s{seed}"]) for seed in RADIO_SEEDS]
    mean = statistics.fmean(labels)
    spread = statistics.pstdev(labels)
    fraction = spread / mean if mean else float("inf")
    layouts = [runs[f"v6_s{seed}"]["background_stations"] for seed in RADIO_SEEDS]
    world_held_still = all(layout == layouts[0] for layout in layouts[1:])

    print(f"\nV6 spread over {len(RADIO_SEEDS)} radio seeds: mean={mean:.2f} "
          f"sd={spread:.2f} sd/mean={fraction:.2f}")
    print(f"    labels: {', '.join(f'{v:.1f}' for v in labels)}")

    passed = (spread > MIN_SEED_SPREAD_MBPS
              and fraction < MAX_SEED_SPREAD_FRACTION
              and world_held_still)
    return Verdict("V6 radio seed varies the label on one fixed world", passed,
                   f"sd={spread:.2f}, sd/mean={fraction:.2f}, "
                   f"stations {'fixed' if world_held_still else 'MOVED'}")


def check_matched_observation(sim: Sim) -> Verdict:
    """V7: every AP the client could join shares one pre-association recording."""
    print("\nV7 matched-set observation identity")
    all_ok = True
    # Three APs can each hold a channel of their own; six cannot, so at six
    # some pair always shares one. That is where a leak between variants
    # listening on the same channel would show up.
    for n_aps in (3, 6):
        fingerprints = []
        for target_ap in range(n_aps):
            tag = f"v7_n{n_aps}_ap{target_ap}"
            sim.run(tag, nAPs=n_aps, nSTAs=n_aps * 3, targetAP=target_ap,
                    candidateX=25, candidateY=17, hotspotAPs="1", bgPerStaMbps=4,
                    rngSeed=RADIO_SEEDS[0])
            frames = (sim.out_dir / tag / "observation.csv")
            channel_busy = (sim.out_dir / tag / "chanbusy.csv")
            if not frames.exists() or not channel_busy.exists():
                raise FileNotFoundError(f"{tag} is missing an observation file")
            frame_bytes = frames.read_bytes()
            busy_bytes = channel_busy.read_bytes()
            if (len(frame_bytes) < MIN_OBSERVATION_BYTES
                    or len(busy_bytes) < MIN_CHANNEL_BUSY_BYTES):
                raise RuntimeError(
                    f"{tag} recorded {len(frame_bytes)} frame bytes and "
                    f"{len(busy_bytes)} channel-busy bytes; too little to compare")
            fingerprints.append((hashlib.md5(frame_bytes).hexdigest()[:10],
                                 hashlib.md5(busy_bytes).hexdigest()[:10]))

        identical = len(set(fingerprints)) == 1
        all_ok = all_ok and identical
        print(f"    nAPs={n_aps}: {'identical' if identical else 'DIFFERENT'} "
              f"across {n_aps} target APs {fingerprints[0]}")

    return Verdict("V7 a client's variants share one observation", all_ok,
                   "frames and channel-busy byte-identical")


def check_shadowing(sim: Sim) -> Verdict:
    """V8: candidateSeed redraws the client's shadowing and touches nothing else."""
    # 30 m is on the steep part of V1's curve, where a few dB of attenuation
    # changes the rate the client can hold. Close in, every draw would saturate.
    scenario = dict(nAPs=1, nSTAs=4, candidateY=30,
                    bgPerStaMbps=IDLE_BACKGROUND_MBPS, rngSeed=RADIO_SEEDS[0])
    jobs = [(f"v8_sigma{sigma}_c{seed}",
             dict(candidateSeed=seed, linkShadowingDb=sigma, **scenario))
            for sigma in (5.0, 0.0) for seed in CANDIDATE_SEEDS]
    runs = sim.run_all(jobs)

    def labels_at(sigma: float) -> list[float]:
        return [throughput(runs[f"v8_sigma{sigma}_c{seed}"]) for seed in CANDIDATE_SEEDS]

    shadowed = labels_at(5.0)
    unshadowed = labels_at(0.0)
    spread = statistics.pstdev(shadowed)

    # Nothing but the client's own attenuation may follow candidateSeed: with no
    # candidateStratum set the client must stand exactly where it was put, and
    # the background stations belong to topologySeed.
    metas = [runs[f"v8_sigma{sigma}_c{seed}"]
             for sigma in (5.0, 0.0) for seed in CANDIDATE_SEEDS]
    only_shadowing_moved = all(
        m["candidate_position"] == metas[0]["candidate_position"]
        and m["candidate_stratum"] == ""
        and m["background_stations"] == metas[0]["background_stations"]
        for m in metas)

    print("\nV8 shadowing drawn from candidateSeed (one AP, client at 30 m)")
    print(f"    sigma=5 dB: {', '.join(f'{v:.2f}' for v in shadowed)}  sd={spread:.2f}")
    print(f"    sigma=0 dB: {', '.join(f'{v:.2f}' for v in unshadowed)}")
    print(f"    client at {metas[0]['candidate_position']}, background fixed: "
          f"{'yes' if only_shadowing_moved else 'NO'}")

    passed = (spread > MIN_SHADOWING_SPREAD_MBPS
              and len(set(unshadowed)) == 1
              and only_shadowing_moved)
    return Verdict("V8 shadowing moves the label, and only the label", passed,
                   f"sd={spread:.2f} Mbps at 5 dB, "
                   f"{len(set(unshadowed))} distinct value(s) at 0 dB")


CHECKS: dict[str, Callable[[Sim], Verdict]] = {
    "V1": check_distance_decay,
    "V2": check_load_decay,
    "V3": check_distance_load_grid,
    "V4": check_layout_and_channels,
    "V5": check_repeatability,
    "V6": check_radio_seed_spread,
    "V7": check_matched_observation,
    "V8": check_shadowing,
}


def main() -> int:
    # A single check is minutes of simulator time, so the progress printed as
    # each one finishes has to reach a redirected log while the suite is still
    # running, not all at once when it exits.
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", required=True,
                        help="the built ns-3 simulator to exercise")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="keep the simulator output here; default is a "
                             "temporary directory deleted on exit")
    parser.add_argument("--workers", type=int, default=8,
                        help="simulator processes to keep running at once")
    parser.add_argument("--repeats", type=int, default=3,
                        help=f"radio seeds averaged per point in V1, V2 and V3 "
                             f"(1 to {len(RADIO_SEEDS)})")
    parser.add_argument("--only", default=None,
                        help="comma-separated checks to run, e.g. V1,V8; "
                             "default runs all of " + ",".join(CHECKS))
    args = parser.parse_args()

    if not 1 <= args.repeats <= len(RADIO_SEEDS):
        parser.error(f"--repeats must be between 1 and {len(RADIO_SEEDS)}")
    selected = list(CHECKS)
    if args.only:
        selected = [name.strip().upper() for name in args.only.split(",")]
        unknown = [name for name in selected if name not in CHECKS]
        if unknown:
            parser.error(f"unknown check(s) {unknown}; choose from {list(CHECKS)}")

    out_dir = args.out_dir or Path(tempfile.mkdtemp(prefix="validate_sim_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    sim = Sim(binary=args.binary, out_dir=out_dir, workers=args.workers,
              radio_seeds=RADIO_SEEDS[:args.repeats])

    verdicts = [CHECKS[name](sim) for name in selected]

    print("\n" + "=" * 72)
    for verdict in verdicts:
        print(f"  [{'PASS' if verdict.passed else 'FAIL'}] "
              f"{verdict.label:<56} {verdict.detail}")
    print("=" * 72)

    if args.out_dir is None:
        shutil.rmtree(out_dir, ignore_errors=True)
    return 0 if all(verdict.passed for verdict in verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
