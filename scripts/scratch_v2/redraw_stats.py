"""Within-scan label spread: futures that keep sessions (f2..f5) vs redraw them (r2..r5)."""
import json, numpy as np
from pathlib import Path
thr = lambda p: json.loads(p.read_text())["candidate"]["throughput_mbps"]
keep, redraw = [], []
for d in sorted(Path("data/runs_v2_redraw").glob("*__r2")):
    tag = d.name[:-4]
    r = [Path("data/runs_v2_redraw")/f"{tag}__r{f}"/"metadata.json" for f in range(2, 6)]
    f = [Path("data/runs_v2")/f"{tag}__f{f}"/"metadata.json" for f in range(2, 6)]
    if all(p.exists() for p in r + f):
        redraw.append([thr(p) for p in r]); keep.append([thr(p) for p in f])
keep, redraw = np.array(keep), np.array(redraw)
def stats(x):
    loo = np.abs(x - (x.sum(1, keepdims=True) - x) / (x.shape[1] - 1))
    med = np.abs(x - np.median(x, 1, keepdims=True))
    return dict(scans=len(x), mad_median=round(med.mean(), 3), loo_mae=round(loo.mean(), 3),
                sd_mean=round(x.std(1, ddof=1).mean(), 3),
                within_var_share=round(x.var(1, ddof=1).mean() / x.var(), 4))
print("keep  ", stats(keep)); print("redraw", stats(redraw))
m = redraw.mean(1); loo = np.abs(redraw - (redraw.sum(1, keepdims=True) - redraw) / 3).mean(1)
for lo, hi in ((0, 1), (1, 10), (10, 30), (30, 50), (50, 61)):
    k = (m >= lo) & (m < hi); print(lo, hi, int(k.sum()), round(loo[k].mean(), 2) if k.any() else None)
