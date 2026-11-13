"""Tabular model given the true link powers of every transmitter on the target AP's channel."""
import argparse, concurrent.futures, json, subprocess, sys, os, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, ".")
from scripts.baselines.regime_analysis import fit_predict, mae, across_folds
from scripts.common.splits import N_FOLDS, split_by_topology
from scripts.simulate.run_sweep import plan_scans
BIN = Path.home() / "workspace/ns-3.45/build/scratch/ns3.45-my-wifi-test"
TX, BANDS = 16.0, ((-82, 1e9), (-92, -82), (-1e9, -92))
t = pd.read_csv("data/aggregate_v2.csv")
t = t[t.topology_id.str[1:].astype(int) < int(sys.argv[1])].reset_index(drop=True)
plans = {s.tag: s for s in plan_scans(argparse.Namespace(n_topologies=1000, sweep_seed=1234))}

def facts(scan_id):
    m = json.loads((Path("data/runs_v2")/f"{scan_id}__draw"/"metadata.json").read_text())
    out = subprocess.run([str(BIN), "--printLossMatrix=1", "--runTag=none", *(f"--{k}={v}" for k, v in plans[scan_id].arguments.items())],
                         capture_output=True, text=True, check=True).stdout
    loss = np.array([[float(x) for x in row.split()] for row in out.strip().split("\n")])
    arguments = [f"--{k}={v}" for k, v in plans[scan_id].arguments.items()]
    lines = subprocess.run([str(BIN), "--printSessionsAtJoin=1", "--captureObs=0", f"--outDir={sys.argv[2]}",
                            "--runTag=x" + scan_id, *arguments], capture_output=True, text=True, check=True).stdout
    active = [int(l.split()[1]) for l in lines.split("\n") if l.startswith("session ")]
    assert len(active) == len(m["background_stations"])
    aps, stas = m["aps"], m["background_stations"]; n_ap = len(aps); cand = loss.shape[0] - 1
    assert loss.shape[0] == n_ap + len(stas) + 1
    assert all(abs(TX - loss[cand, a["index"]] - a["candidate_rx_dbm"]) < 0.01 for a in aps)
    rows = []
    for ap in aps:
        r = {"scan_id": scan_id, "ap_index": ap["index"], "l_rx": TX - loss[cand, ap["index"]]}
        for c in range(3):
            for a in range(4):
                r[f"l_load_c{c}_a{a}"] = 0.0; r[f"l_n_c{c}_a{a}"] = 0; r[f"s_on_c{c}_a{a}"] = 0.0; r[f"s_n_c{c}_a{a}"] = 0
        for s in stas:
            if aps[s["ap"]]["channel"] != ap["channel"]: continue
            node = s["ap"] if s["is_downlink"] else n_ap + s["index"]
            load = s["offered_mbps"] * s["mean_session_s"] / (s["mean_session_s"] + s["mean_pause_s"])
            c = next(i for i, (lo, hi) in enumerate(BANDS) if lo <= TX - loss[node, cand] < hi)
            a = 3 if node == ap["index"] else next(i for i, (lo, hi) in enumerate(BANDS) if lo <= TX - loss[node, ap["index"]] < hi)
            r[f"l_load_c{c}_a{a}"] += load; r[f"l_n_c{c}_a{a}"] += 1
            r[f"s_on_c{c}_a{a}"] += s["offered_mbps"] * active[s["index"]]; r[f"s_n_c{c}_a{a}"] += active[s["index"]]
        rows.append(r)
    return rows

with concurrent.futures.ThreadPoolExecutor(4) as pool:
    rows = [r for scan in pool.map(facts, t.scan_id.unique()) for r in scan]
k = t.merge(pd.DataFrame(rows), on=["scan_id", "ap_index"], validate="1:1")
feat = [c for c in t.columns if c.startswith("feat_")]; orc = [c for c in k.columns if c.startswith("l_")]; ses = [c for c in k.columns if c.startswith("s_")]
print(len(k), "rows", k.scan_id.nunique(), "scans", flush=True)
for name, cols in (("features", feat), ("link_oracle", orc), ("features_links", feat + orc), ("sessions_links", orc + ses), ("features_links_sessions", feat + orc + ses)):
    e = []
    for f in range(N_FOLDS):
        tr, va, te = split_by_topology(k, test_fold=f)
        e.append(mae(te, fit_predict(pd.concat([tr, va]), te, cols)))
    print(name, len(cols), across_folds(e), flush=True)
