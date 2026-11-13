"""Tabular model given a rich description of the true deployment around the target AP."""
import json, sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, ".")
from scripts.baselines.regime_analysis import fit_predict, mae, across_folds
from scripts.common.splits import N_FOLDS, split_by_topology
t = pd.read_csv("data/aggregate_v2.csv")
rows = []
for scan_id, g in t.groupby("scan_id"):
    m = json.loads((Path("data/runs_v2")/f"{scan_id}__draw"/"metadata.json").read_text())
    c = m["candidate_position"]; aps = {a["index"]: a for a in m["aps"]}
    for ap_index in g.ap_index:
        ap = aps[ap_index]; r = {"scan_id": scan_id, "ap_index": ap_index,
            "o_rx": ap["candidate_rx_dbm"],
            "o_dist": np.hypot(ap["position"]["x"]-c["x"], ap["position"]["y"]-c["y"])}
        for name, own in (("own", True), ("co", False)):
            stas = [s for s in m["background_stations"]
                    if aps[s["ap"]]["channel"] == ap["channel"] and ((s["ap"] == ap_index) == own)]
            for band, lo, hi in (("near", 0, 10), ("mid", 10, 25), ("far", 25, 1e9)):
                sel = [s for s in stas if lo <= np.hypot(s["position"]["x"]-c["x"], s["position"]["y"]-c["y"]) < hi]
                duty = [s["mean_session_s"]/(s["mean_session_s"]+s["mean_pause_s"]) for s in sel]
                off = [s["offered_mbps"] for s in sel]
                k = f"o_{name}_{band}_"
                r[k+"n"] = len(sel); r[k+"load"] = sum(o*d for o, d in zip(off, duty))
                r[k+"peak"] = sum(off); r[k+"up"] = sum(o*d for o, d, s in zip(off, duty, sel) if not s["is_downlink"])
                r[k+"tcp"] = sum(o*d for o, d, s in zip(off, duty, sel) if s["is_tcp"])
                r[k+"slow"] = sum(o*d for o, d, s in zip(off, duty, sel) if s["mean_session_s"] > 5)
        rows.append(r)
k = t.merge(pd.DataFrame(rows), on=["scan_id", "ap_index"], validate="1:1")
feat = [c for c in t.columns if c.startswith("feat_")]; orc = [c for c in k.columns if c.startswith("o_")]
for name, cols in (("features", feat), ("oracle", orc), ("both", feat+orc)):
    e = []
    for f in range(N_FOLDS):
        tr, va, te = split_by_topology(k, test_fold=f)
        e.append(mae(te, fit_predict(pd.concat([tr, va]), te, cols)))
    print(name, len(cols), across_folds(e), flush=True)
