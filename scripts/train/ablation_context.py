#!/usr/bin/env python3
"""Does the set context earn its keep, or do the relative features do its job?

The choice-set model can learn to compare options in two quite different
ways, and they overlap:

  * feat_rel_* columns hand-compute the comparison (RSSI rank, margin over
    the best other AP, share of observed airtime), so even a model that
    scores each option in isolation is handed the context.
  * the set encoder pools across the options and conditions each score on
    that pooling, learning the comparison instead.

This runs the full 2x2 needed to separate them: set context on/off, relative
features present/removed. The interesting cell is bottom-left - no relative
features, context on - which shows whether the architecture can recover by
itself what the hand engineering supplies.

Run:  python scripts/ablation_context.py data/dataset.csv --repeats 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.data import feature_columns, impute_features, load_dataset, split_by_group  # noqa: E402
from models.evaluate import selection_metrics  # noqa: E402
from models.ranker import (SetRanker, Standardizer, make_groups, predict_rows,  # noqa: E402
                           train as train_ranker)


def run_cell(df, feats, use_context, seed, epochs, patience):
    train, val, test = split_by_group(df, seed=seed)
    scaler = Standardizer(train[feats].to_numpy(dtype=np.float32))
    tr, va, te = (make_groups(d, feats, scaler) for d in (train, val, test))

    best = ((np.inf, np.inf), None)
    for alpha, temp in ((0.3, 5.0), (0.7, 2.0), (0.7, 5.0),
                        (0.7, 10.0), (0.9, 5.0)):
        torch.manual_seed(seed)
        m = SetRanker(len(feats), dropout=0.1, use_context=use_context)
        m = train_ranker(m, tr, va, epochs=epochs, lr=3e-3, weight_decay=1e-3,
                         alpha=alpha, temperature=temp, seed=seed, verbose=False,
                         patience=patience)
        p = np.expm1(np.clip(predict_rows(m, va, len(val)), -5, 12))
        sm = selection_metrics(val, p)
        regret = sm.get("topology_mean_regret_mbps", sm["mean_regret_mbps"])
        key = (regret, -sm["mean_spearman"])
        if key < best[0]:
            best = (key, m)
    model = best[1]
    p_te = np.expm1(np.clip(predict_rows(model, te, len(test)), -5, 12))
    return selection_metrics(test, p_te)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--out", type=Path, default=Path("results/ablation_context.json"))
    args = parser.parse_args()

    df = impute_features(load_dataset(args.dataset))
    all_feats = feature_columns(df)
    no_rel = [c for c in all_feats if not c.startswith("feat_rel_")]
    print(f"features: {len(all_feats)} total, {len(no_rel)} without feat_rel_*\n")

    rows = []
    for rel_name, feats in (("with feat_rel_*", all_feats), ("without feat_rel_*", no_rel)):
        for ctx_name, use_ctx in (("set context", True), ("pointwise", False)):
            regrets, top1s = [], []
            for seed in range(args.repeats):
                m = run_cell(df, feats, use_ctx, seed, args.epochs, args.patience)
                regrets.append(m.get("topology_mean_regret_mbps",
                                     m["mean_regret_mbps"]))
                top1s.append(m["top1_accuracy"])
            rows.append({
                "relative_features": rel_name, "architecture": ctx_name,
                "regret_mean": float(np.mean(regrets)), "regret_std": float(np.std(regrets)),
                "top1_mean": float(np.mean(top1s)),
            })
            print(f"{rel_name:<20} {ctx_name:<13} "
                  f"regret={np.mean(regrets):.3f} +/- {np.std(regrets):.3f}  "
                  f"top1={np.mean(top1s):.3f}")

    res = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))

    piv = res.pivot(index="relative_features", columns="architecture", values="regret_mean")
    print("\nmean regret (Mbps), lower is better:")
    print(piv.to_string())
    tex = piv.to_latex(float_format="%.3f",
                       caption="Ablation: mean test regret (Mbps) with and without the "
                               "hand-computed relative features, for the set-context "
                               "encoder and its pointwise equivalent. Both architectures "
                               "have identical parameter counts.",
                       label="tab:ablation")
    args.out.with_suffix(".tex").write_text(tex)
    print(f"\nwrote {args.out} and {args.out.with_suffix('.tex')}")


if __name__ == "__main__":
    main()
