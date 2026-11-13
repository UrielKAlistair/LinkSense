"""Replicated scans re-run with the sessions redrawn at the join (scratch driver).
Usage: redraw_floor.py <out_dir> <workers> [<n_scans>]"""
import argparse, concurrent.futures, json, sys
from pathlib import Path
sys.path.insert(0, ".")
from scripts.simulate.run_sweep import plan_scans, simulate

out_dir, workers = Path(sys.argv[1]), int(sys.argv[2])
limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
args = argparse.Namespace(binary=Path.home() / "workspace/ns-3.45/build/scratch/ns3.45-my-wifi-test",
                          out_dir=out_dir, n_topologies=1000, sweep_seed=1234, timeout=1800.0)
out_dir.mkdir(parents=True, exist_ok=True)
scans = [s for s in plan_scans(args) if s.is_replicated][:limit]

def run(scan):
    target = json.loads((Path("data/runs_v2") / f"{scan.tag}__draw" / "metadata.json").read_text())["candidate"]["target_ap"]
    return [simulate(args, f"{scan.tag}__r{f}", scan.arguments,
                     {"targetAP": target, "futureSeed": f, "redrawSessions": 1, "captureObs": 0})
            for f in range(2, 6)]

with concurrent.futures.ThreadPoolExecutor(workers) as pool:
    for done, outcomes in enumerate(pool.map(run, scans), 1):
        if done % 20 == 0 or done == len(scans):
            print(done, len(scans), sum(o is None for o in outcomes), flush=True)
