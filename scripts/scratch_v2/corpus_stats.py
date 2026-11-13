"""Label, regime and noise-floor statistics of a v2 sweep, read from metadata.json."""
import json, glob, sys, collections
import numpy as np
runs = sys.argv[1]
draw, fut, matched = {}, collections.defaultdict(list), collections.defaultdict(dict)
for f in glob.glob(f"{runs}/*/metadata.json"):
    name = f.split("/")[-2]; scan, kind = name[:11], name[13:]
    m = json.load(open(f)); c = m["candidate"]
    if kind == "draw": draw[scan] = m
    elif kind.startswith("f"): fut[scan].append(c["throughput_mbps"])
    else: matched[scan][c["target_ap"]] = c["throughput_mbps"]
T = np.array([m["candidate"]["throughput_mbps"] for m in draw.values()])
A = np.array([m["candidate"]["associated"] for m in draw.values()])
print(f"{len(T)} scans; never associated {1-A.mean():.3f}; label==0 {np.mean(T==0):.3f}; <1 {np.mean(T<1):.3f}")
print("label quantiles 10/25/50/75/90:", np.quantile(T,[.1,.25,.5,.75,.9]).round(2), "mean", T.mean().round(2), "sd", T.std().round(2))
print("hist 0,1,5,10,20,30,40,50,60:", np.histogram(T, bins=[0,1,5,10,20,30,40,50,60])[0])
print("predict-median MAE", np.abs(T-np.median(T)).mean().round(2))
# noise floor
dev, sds, rows = [], [], []
for scan, v in fut.items():
    v = np.array(v + [draw[scan]["candidate"]["throughput_mbps"]])
    if len(v) == 5:
        # leave-one-out: predict each future by the mean of the other four
        loo = np.abs(v - (v.sum() - v) / 4).mean()
        dev.append(np.abs(v - np.median(v)).mean()); sds.append(v.std(ddof=1)); rows.append((v.mean(), loo))
rows = np.array(rows)
print(f"replicated scans {len(dev)}: MAD from scan median {np.mean(dev):.3f}; within-scan sd mean {np.mean(sds):.3f} median {np.median(sds):.3f}; leave-one-out MAE {rows[:,1].mean():.3f}")
allv = np.concatenate([np.array(v + [draw[s]['candidate']['throughput_mbps']]) for s, v in fut.items() if len(v)==4])
within = np.mean(np.square(sds)); total = allv.var()
print(f"within-scan variance share {within/total:.4f} (total sd {np.sqrt(total):.2f})")
for lo, hi in [(0,1),(1,10),(10,30),(30,50),(50,61)]:
    k = (rows[:,0]>=lo)&(rows[:,0]<hi)
    print(f"  mean label {lo}-{hi}: n={k.sum()} MAD-like LOO MAE {rows[k,1].mean():.2f}")
# matched: is strongest-RSSI the best, oracle gain
gain, n = [], 0
for scan, others in matched.items():
    m = draw[scan]; labels = dict(others); labels[m["candidate"]["target_ap"]] = m["candidate"]["throughput_mbps"]
    rx = {a["index"]: a["candidate_rx_dbm"] for a in m["aps"]}
    best = max(labels.values()); strongest = labels[max(labels, key=lambda a: rx[a])]
    gain.append(best - strongest); n += 1
gain = np.array(gain)
print(f"matched scans {n}: strongest-power AP is best in {np.mean(gain<=1e-9):.3f}; its mean regret {gain.mean():.2f} Mbit/s")
# decomposition
d = [m["candidate"] for m in draw.values() if m["candidate"]["associated"] and m["candidate"]["data_frames_sent"]>0]
air = np.array([c["transmit_seconds"]/c["observed_seconds"] for c in d]); rate = np.array([c["mean_phy_rate_mbps"] for c in d])
deliv = np.array([c["data_frames_acked"]/c["data_frames_sent"] for c in d]); thr = np.array([c["throughput_mbps"] for c in d])
for nm, x in [("airtime share", air), ("phy rate", rate), ("delivery ratio", deliv)]:
    print(f"  {nm}: quartiles {np.quantile(x,[.25,.5,.75]).round(2)}, var(log) {np.var(np.log(np.clip(x,1e-3,None))):.2f}")
k = thr > 0.1
prod = air*rate*deliv
print("  corr(log thr, log product)", np.corrcoef(np.log(thr[k]), np.log(prod[k]))[0,1].round(3), "median thr/product", np.median(thr[k]/prod[k]).round(3))
# temporal variability
cv = []
for c in d:
    b = np.array(c["bytes_per_second"][1:], float)
    if b.mean() > 0: cv.append(b.std()/b.mean())
print("  per-second CV of delivered rate: quartiles", np.quantile(cv,[.25,.5,.75]).round(2))
