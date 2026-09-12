#!/usr/bin/env python3
"""Does letting the model see the rival access points earn its keep?

Choosing an access point is a comparison, so a model has to get the comparison
from somewhere. There are two places it can come from, and this script exists
because they overlap.

The first is the feature table. Some of its columns are already comparative -
this AP's rank by signal strength among the ones on offer, how far its signal
sits above the next best, its share of the airtime being used. A model that
looks at one option at a time is still handed the comparison, precomputed, in
those columns.

The second is the architecture. The set encoder pools across all the options in
a scan and conditions each option's score on that pool, so it can work
the comparison out for itself.

Give a model both and neither gets the credit. So this runs all four
combinations: comparative columns kept or dropped, set encoder on or off. The
cell that answers the question is the one with the columns dropped and the
encoder on - if it holds up, the architecture recovers on its own what the hand
engineering was supplying, and the hand engineering can go.

The two architectures have the same number of parameters: switching the encoder
off feeds zeros into the same score head rather than making a smaller network,
so a difference between them cannot be a difference in capacity.

Run:
  .venv/bin/python3 scripts/train/ablation_context.py data/v3_dataset.csv \
      --out results_v3/ablation_context.json
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

from models.data import feature_columns, impute_features, load_dataset, split_by_topology  # noqa: E402
from models.evaluate import selection_metrics, validation_selection_key  # noqa: E402
from models.ranker import (SetRanker, Standardizer, pack_scans, predict_rows,  # noqa: E402
                           train as train_ranker)
from scripts.train.train_eval import RANKER_OBJECTIVES  # noqa: E402


def run_cell(df, feats, use_context, seed, epochs, patience):
    """Train one cell of the 2x2 and return its test-set decision metrics."""
    train, val, test = split_by_topology(df, seed=seed)
    scaler = Standardizer(train[feats].to_numpy(dtype=np.float32))
    tr, va, te = (pack_scans(d, feats, scaler) for d in (train, val, test))

    best = ((np.inf, np.inf), None)
    # The same five objective settings scripts/train/train_eval.py searches, so
    # the two cells that both scripts measure stay comparable.
    for alpha, temp in RANKER_OBJECTIVES:
        torch.manual_seed(seed)
        m = SetRanker(len(feats), dropout=0.1, use_context=use_context)
        m = train_ranker(m, tr, va, epochs=epochs, lr=3e-3, weight_decay=1e-3,
                         alpha=alpha, temperature=temp, seed=seed, verbose=False,
                         patience=patience)
        p = np.expm1(np.clip(predict_rows(m, va, len(val)), -5, 12))
        key = validation_selection_key(val, p)
        if key < best[0]:
            best = (key, m)
    model = best[1]
    p_te = np.expm1(np.clip(predict_rows(model, te, len(test)), -5, 12))
    return selection_metrics(test, p_te)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--repeats", type=int, default=5,
                        help="independent topology splits; matches train_eval.py so "
                             "the shared cells can be compared row for row")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
