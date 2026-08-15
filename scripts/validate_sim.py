#!/usr/bin/env python3
"""Controlled experiments that check the simulator behaves like real WiFi.

This is not a unit test suite - it is a set of physics/behaviour sanity
checks whose *shape* we can reason about independently of the simulator.
If any of these come out non-monotone or flat, something is wrong in the
sim and the resulting dataset would teach a model nonsense.

Experiments:
  V1 link-quality: throughput vs distance, no contention.
     expect: high and flat when close, smooth monotone decay, ~0 when far.
  V2 load: throughput vs background offered load, distance fixed.
     expect: monotone decreasing.
  V3 interaction: throughput vs distance at several background loads.
     expect: V1's curve shifted down as load rises.
  V4 channels: 2 APs co-channel vs on separate channels.
     expect: separate channels >= co-channel (no cross-BSS contention).
  V5 determinism: same seed twice => byte-identical label.
  V6 seed spread: same scenario, different seeds => variation, but not wild.

Run:  python scripts/validate_sim.py --binary <path> [--out-dir <dir>]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

BASE = {
    "simStopTime": 14.0,
    "candidateStartTime": 6.0,
    "jitterStd": 0.0,
    "candidateAngleDeg": 90.0,
}


def run(binary: str, out_dir: Path, tag: str, **kwargs) -> dict:
    args = [binary, f"--outDir={out_dir}", f"--runTag={tag}"]
    params = {**BASE, **kwargs}
    for k, v in params.items():
        args.append(f"--{k}={v}")
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"run failed: {' '.join(args)}\n{proc.stderr[-1500:]}")
    return json.loads((out_dir / tag / "metadata.json").read_text())


def parallel(binary: str, out_dir: Path, jobs: list[tuple[str, dict]], workers: int) -> dict[str, dict]:
    results: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run, binary, out_dir, tag, **kw): tag for tag, kw in jobs}
        for fut in concurrent.futures.as_completed(futs):
            results[futs[fut]] = fut.result()
    return results


def thr(meta: dict) -> float:
    return meta["candidate"]["throughput_mbps"]


def check_monotone_decreasing(values: list[float], tol: float) -> bool:
    """Allow small non-monotonicity: fading and backoff are stochastic, so
    a strict test would fail on noise rather than on real problems."""
    return all(b <= a + tol for a, b in zip(values, values[1:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    tmp = args.out_dir or Path(tempfile.mkdtemp(prefix="validate_sim_"))
    tmp.mkdir(parents=True, exist_ok=True)
    seeds = [11 + 7 * i for i in range(args.seeds)]
    verdicts: list[tuple[str, bool, str]] = []

    # ---- V1: link quality vs distance, contention removed ----
    dists = [1, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60]
    jobs = [(f"v1_d{d}_s{s}", dict(nAPs=1, nSTAs=1, candidateDistance=d,
                                  bgPerStaMbps=0.05, rngSeed=s))
            for d in dists for s in seeds]
    res = parallel(args.binary, tmp, jobs, args.workers)
    v1 = [statistics.fmean(thr(res[f"v1_d{d}_s{s}"]) for s in seeds) for d in dists]
    print("V1 link quality vs distance (no contention)")
    for d, t in zip(dists, v1):
        print(f"    {d:>4} m  {t:>7.2f} Mbps  {'#' * int(t / 1.5)}")
    ok = check_monotone_decreasing(v1, tol=2.0) and v1[0] > 30 and v1[-1] < 5
    verdicts.append(("V1 monotone decay, high near / ~0 far", ok, f"{v1[0]:.1f} -> {v1[-1]:.1f} Mbps"))

    # ---- V2: throughput vs background load, distance fixed ----
    loads = [0.05, 2, 4, 6, 8, 10, 12]
    jobs = [(f"v2_l{l}_s{s}", dict(nAPs=1, nSTAs=4, candidateDistance=10,
                                  bgPerStaMbps=l, rngSeed=s))
            for l in loads for s in seeds]
    res = parallel(args.binary, tmp, jobs, args.workers)
    v2 = [statistics.fmean(thr(res[f"v2_l{l}_s{s}"]) for s in seeds) for l in loads]
    print("\nV2 throughput vs background load (4 STAs, candidate at 10 m)")
    for l, t in zip(loads, v2):
        print(f"    {l:>5} Mbps/STA ({l * 4:>5.1f} total)  {t:>7.2f} Mbps  {'#' * int(t / 1.5)}")
    ok = check_monotone_decreasing(v2, tol=3.0) and v2[0] > v2[-1]
    verdicts.append(("V2 load reduces candidate throughput", ok, f"{v2[0]:.1f} -> {v2[-1]:.1f} Mbps"))

    # ---- V3: distance x load interaction ----
    print("\nV3 distance x background load")
    d3 = [5, 15, 25, 35]
    l3 = [0.05, 4, 8]
    jobs = [(f"v3_d{d}_l{l}_s{s}", dict(nAPs=1, nSTAs=4, candidateDistance=d,
                                       bgPerStaMbps=l, rngSeed=s))
            for d in d3 for l in l3 for s in seeds]
    res = parallel(args.binary, tmp, jobs, args.workers)
    print(f"    {'dist':>6}" + "".join(f"{f'{l}Mbps':>11}" for l in l3))
    grid = {}
    for d in d3:
        row = [statistics.fmean(thr(res[f"v3_d{d}_l{l}_s{s}"]) for s in seeds) for l in l3]
        grid[d] = row
        print(f"    {d:>6}" + "".join(f"{v:>11.2f}" for v in row))
    ok = all(check_monotone_decreasing(grid[d], tol=3.0) for d in d3) and \
         check_monotone_decreasing([grid[d][0] for d in d3], tol=3.0)
    verdicts.append(("V3 both axes degrade throughput", ok, "see grid"))

    # ---- V4: co-channel vs separate channels ----
    jobs = []
    for nch in (1, 2):
        for s in seeds:
            jobs.append((f"v4_c{nch}_s{s}", dict(nAPs=2, nSTAs=8, apSpacing=25,
                                                candidateDistance=8, nChannels=nch,
                                                bgPerStaMbps=6, rngSeed=s)))
    res = parallel(args.binary, tmp, jobs, args.workers)
    co = statistics.fmean(thr(res[f"v4_c1_s{s}"]) for s in seeds)
    sep = statistics.fmean(thr(res[f"v4_c2_s{s}"]) for s in seeds)
    print(f"\nV4 co-channel={co:.2f} Mbps   separate-channels={sep:.2f} Mbps")
    verdicts.append(("V4 channel separation helps", sep > co, f"{co:.1f} -> {sep:.1f} Mbps"))

    # ---- V5: determinism ----
    a = run(args.binary, tmp, "v5_a", nAPs=2, nSTAs=6, candidateDistance=12, rngSeed=99)
    b = run(args.binary, tmp, "v5_b", nAPs=2, nSTAs=6, candidateDistance=12, rngSeed=99)
    same = thr(a) == thr(b)
    print(f"\nV5 determinism: {thr(a):.6f} vs {thr(b):.6f} -> {'identical' if same else 'DIFFERENT'}")
    verdicts.append(("V5 same seed reproduces exactly", same, f"{thr(a):.4f}"))

    # ---- V6: seed spread ----
    many = [11 + 3 * i for i in range(8)]
    jobs = [(f"v6_s{s}", dict(nAPs=2, nSTAs=6, candidateDistance=18, bgPerStaMbps=6, rngSeed=s))
            for s in many]
    res = parallel(args.binary, tmp, jobs, args.workers)
    vals = [thr(res[f"v6_s{s}"]) for s in many]
    mean = statistics.fmean(vals)
    sd = statistics.pstdev(vals)
    print(f"\nV6 seed spread at fixed scenario: mean={mean:.2f} sd={sd:.2f} "
          f"cv={sd / mean if mean else float('nan'):.2f}")
    print(f"    values: {', '.join(f'{v:.1f}' for v in vals)}")
    verdicts.append(("V6 seeds vary but not degenerately", sd > 0.01, f"sd={sd:.2f}"))

    # ---- V7: matched-set observation identity ----
    # The choice-set framing depends on every variant of a group sharing one
    # pre-association observation. Verify it at the bytes rather than
    # assuming it, on both a co-channel and a multi-channel deployment.
    print("\nV7 matched-set observation identity")
    v7_ok = True
    for nch in (1, 3):
        digests = []
        for t in range(3):
            meta = run(args.binary, tmp, f"v7_c{nch}_t{t}", nAPs=3, nSTAs=9, apSpacing=30,
                      targetAP=t, candidateAbsolute=1, candidateX=35, candidateY=12,
                      nChannels=nch, bgPerStaMbps=4, rngSeed=4242)
            obs = tmp / f"v7_c{nch}_t{t}" / "observation.csv"
            if not obs.exists():
                raise FileNotFoundError(f"{obs} missing; V7 cannot verify anything")
            body = obs.read_bytes()
            digests.append((hashlib.md5(body).hexdigest()[:10], len(body)))
        # guard against the check passing because nothing was compared
        assert all(d[1] > 1000 for d in digests), "observation files suspiciously small"
        same = len(set(digests)) == 1
        v7_ok &= same
        print(f"    nChannels={nch}: {'identical' if same else 'DIFFERENT'} across 3 variants "
              f"({digests[0]})")
    verdicts.append(("V7 group variants share one observation", v7_ok, "byte-identical scans"))

    print("\n" + "=" * 68)
    for name, ok, detail in verdicts:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<42} {detail}")
    print("=" * 68)

    if args.out_dir is None:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if all(ok for _, ok, _ in verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
