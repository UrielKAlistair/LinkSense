#!/usr/bin/env python3
"""Check that every run of a scan hears the same thing before the candidate joins.

A scan is one deployment and one candidate position, simulated once per target
AP. This draws scans with run_sweep.py's sampler and, for each:

  1. runs every target AP with the observation recorded, and requires
     observation.csv and chanbusy.csv to be byte-identical across them and to
     hold at least one frame;
  2. reruns the last target AP without recording, and requires its
     metadata.json to be byte-identical to the recorded run's.

Prints one line per scan and exits non-zero if any scan fails.

Run:  .venv/bin/python scripts/tests/check_observation_identity.py --binary <path>
"""

from __future__ import annotations

import argparse
import concurrent.futures
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.simulate.run_sweep import sample_candidate, sample_topology  # noqa: E402


def plan_scans(n_scans: int, seed: int) -> list[dict]:
    """Simulator settings for each scan, drawn with run_sweep.py's sampler."""
    scans = []
    for index in range(n_scans):
        rng = random.Random(f"identity:{seed}:{index}")
        scans.append({**sample_topology(rng), **sample_candidate(rng),
                      "rngSeed": rng.randrange(1, 2**31 - 1)})
    return scans


def simulate(binary: str, out_dir: Path, tag: str, settings: dict) -> None:
    """Run the simulator once into out_dir/tag."""
    argv = [binary, f"--outDir={out_dir}", f"--runTag={tag}"]
    argv += [f"--{name}={value}" for name, value in settings.items()]
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"run failed: {' '.join(argv)}\n{proc.stderr[-1500:]}")


def identical(paths: list[Path]) -> bool:
    first = paths[0].read_bytes()
    return all(path.read_bytes() == first for path in paths[1:])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--n-scans", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0,
                        help="which scans are drawn")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="keep the runs here; default is a temporary "
                             "directory deleted on exit")
    args = parser.parse_args()

    out_dir = args.out_dir or Path(tempfile.mkdtemp(prefix="observation_identity_"))
    recorded_dir = out_dir / "recorded"
    unrecorded_dir = out_dir / "unrecorded"
    recorded_dir.mkdir(parents=True, exist_ok=True)
    unrecorded_dir.mkdir(parents=True, exist_ok=True)

    try:
        scans = plan_scans(args.n_scans, args.seed)
        jobs = []
        for index, scan in enumerate(scans):
            last_ap = scan["nAPs"] - 1
            for target_ap in range(scan["nAPs"]):
                jobs.append((recorded_dir, f"s{index:02d}__ap{target_ap}",
                             {**scan, "targetAP": target_ap, "captureObs": 1}))
            jobs.append((unrecorded_dir, f"s{index:02d}__ap{last_ap}",
                         {**scan, "targetAP": last_ap, "captureObs": 0}))

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in [pool.submit(simulate, args.binary, *job) for job in jobs]:
                future.result()

        all_passed = True
        for index, scan in enumerate(scans):
            runs = [recorded_dir / f"s{index:02d}__ap{ap}" for ap in range(scan["nAPs"])]
            n_frames = len((runs[0] / "observation.csv").read_text().splitlines()) - 1
            observation_same = identical([run / "observation.csv" for run in runs])
            busy_same = identical([run / "chanbusy.csv" for run in runs])
            metadata_same = identical([runs[-1] / "metadata.json",
                                       unrecorded_dir / runs[-1].name / "metadata.json"])
            passed = n_frames > 0 and observation_same and busy_same and metadata_same
            all_passed = all_passed and passed
            print(f"[{'PASS' if passed else 'FAIL'}] scan {index}: "
                  f"nAPs={scan['nAPs']} nSTAs={scan['nSTAs']} frames={n_frames}  "
                  f"observation {'same' if observation_same else 'DIFFERENT'}, "
                  f"chanbusy {'same' if busy_same else 'DIFFERENT'}, "
                  f"metadata without recording {'same' if metadata_same else 'DIFFERENT'}")
        return 0 if all_passed else 1
    finally:
        if args.out_dir is None:
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
