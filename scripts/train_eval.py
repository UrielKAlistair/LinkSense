#!/usr/bin/env python3
"""Train every model, select on validation, report on held-out test groups.

Protocol, kept deliberately strict because the dataset is small enough that
sloppiness here would dominate the results:

  * splits are by GROUP, so no scenario contributes rows to two splits
  * hyperparameters are chosen on validation only, never on test
  * the whole protocol is repeated over several split seeds and reported as
    mean +/- std; a single split of ~160 test groups is noisy enough that
    one number would be misleading
  * every learned model is compared against the heuristics a real client
    could run instead - a model that cannot beat strongest-RSSI has not
    justified itself

Outputs a results table (CSV + LaTeX) and figures for the report.

Run:  python scripts/train_eval.py data/dataset.csv --out-dir report/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.data import (assert_no_leakage, feature_columns, impute_features,  # noqa: E402
                         load_dataset, split_by_group, to_xy)
from models.evaluate import evaluate_all, oracle_ceiling, selection_metrics  # noqa: E402

METRICS = ["top1_accuracy", "mean_regret_mbps", "median_regret_mbps",
           "mean_regret_frac", "mean_spearman", "mae", "rmse", "r2"]


def _select(candidates, val):
    """Pick by validation regret, breaking ties on within-group ordering.

    Regret bottoms out at zero as soon as a configuration gets every
    validation group right, and on a few hundred groups several will. Left
    unbroken, the tie would be settled by grid order rather than evidence.
    """
    best = ((np.inf, np.inf), None)
    for key_extra, pred in candidates:
        sm = selection_metrics(val, pred)
        key = (sm["mean_regret_mbps"], -sm["mean_spearman"])
        if key < best[0]:
            best = (key, key_extra)
    return best


def fit_tree_models(train, val, test, feats, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

    X_tr, y_tr = to_xy(train, feats)
    X_va, _ = to_xy(val, feats)
    X_te, _ = to_xy(test, feats)

    grids = {
        "hist_gbr": (HistGradientBoostingRegressor, [
            dict(max_iter=n, learning_rate=lr, min_samples_leaf=leaf,
                 l2_regularization=1.0, early_stopping=False, random_state=seed)
            for n in (200, 400) for lr in (0.03, 0.06, 0.1) for leaf in (4, 8, 16)
        ]),
        "random_forest": (RandomForestRegressor, [
            dict(n_estimators=400, min_samples_leaf=leaf, max_features=mf,
                 n_jobs=-1, random_state=seed)
            for leaf in (1, 2, 4) for mf in ("sqrt", 0.5, 1.0)
        ]),
    }

    val_pred, test_pred, chosen = {}, {}, {}
    for name, (ctor, grid) in grids.items():
        cands = []
        for log_target in (False, True):
            for cfg in grid:
                m = ctor(**cfg)
                m.fit(X_tr, np.log1p(y_tr) if log_target else y_tr)
                p = m.predict(X_va)
                p = np.expm1(np.clip(p, -5, 12)) if log_target else p
                cands.append(((cfg, log_target), p))
        (regret, _), (cfg, log_target) = _select(cands, val)

        m = ctor(**cfg)
        m.fit(X_tr, np.log1p(y_tr) if log_target else y_tr)
        for X, store in ((X_va, val_pred), (X_te, test_pred)):
            p = m.predict(X)
            store[name] = np.expm1(np.clip(p, -5, 12)) if log_target else p
        chosen[name] = {k: v for k, v in cfg.items() if k not in ("random_state", "n_jobs")}
        chosen[name].update(log_target=log_target, val_regret=round(regret, 4))
    return val_pred, test_pred, chosen


def fit_ranker(train, val, test, feats, seed):
    import torch

    from models.ranker import (SetRanker, Standardizer, make_groups, predict_rows,
                               train as train_ranker)

    scaler = Standardizer(train[feats].to_numpy(dtype=np.float32))
    tr, va, te = (make_groups(d, feats, scaler) for d in (train, val, test))

    cands, models = [], {}
    # alpha stays below 1 so the pointwise term is always present: a purely
    # listwise score is defined only up to a monotone transform, which leaves
    # ranking intact but makes the regression metrics meaningless.
    for alpha in (0.3, 0.5, 0.7, 0.9):
        for temp in (2.0, 5.0, 10.0):
            torch.manual_seed(seed)
            m = SetRanker(len(feats), dropout=0.1)
            m = train_ranker(m, tr, va, epochs=300, lr=3e-3, weight_decay=1e-3,
                             alpha=alpha, temperature=temp, seed=seed, verbose=False)
            p = np.expm1(np.clip(predict_rows(m, va, len(val)), -5, 12))
            cands.append(((alpha, temp), p))
            models[(alpha, temp)] = m
    (regret, _), (alpha, temp) = _select(cands, val)
    model = models[(alpha, temp)]

    return (
        {"set_ranker": np.expm1(np.clip(predict_rows(model, va, len(val)), -5, 12))},
        {"set_ranker": np.expm1(np.clip(predict_rows(model, te, len(test)), -5, 12))},
        {"set_ranker": {"alpha": alpha, "temperature": temp,
                        "val_regret": round(regret, 4)}},
    )


def run_once(df, feats, seed, skip_ranker):
    train, val, test = split_by_group(df, seed=seed)
    val_pred, test_pred, chosen = fit_tree_models(train, val, test, feats, seed)
    if not skip_ranker:
        v, t, c = fit_ranker(train, val, test, feats, seed)
        val_pred.update(v)
        test_pred.update(t)
        chosen.update(c)
    res = evaluate_all(test, test_pred)
    res["split_seed"] = seed
    return res, chosen, test, test_pred


def make_figures(test, preds, agg, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    learned = set(preds)
    order = agg.sort_values(("mean_regret_mbps", "mean"))
    labels = order.index.tolist()
    means = order[("mean_regret_mbps", "mean")].to_numpy()
    errs = np.nan_to_num(order[("mean_regret_mbps", "std")].to_numpy())

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.barh(labels, means, xerr=errs,
            color=["#c44e52" if m in learned else "#8c8c8c" for m in labels],
            error_kw=dict(ecolor="#333333", lw=0.9, capsize=3))
    ax.set_xlabel("mean regret vs oracle (Mbps, lower is better)")
    ax.set_title("Cost of choosing the wrong AP")
    ax.invert_yaxis()
    for y, v in enumerate(means):
        ax.text(v + max(means) * 0.01, y, f"{v:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "regret.pdf")
    plt.close(fig)

    best_model = min(preds, key=lambda k: selection_metrics(test, preds[k])["mean_regret_mbps"])
    y = test["label_throughput_mbps"].to_numpy()
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.scatter(y, preds[best_model], s=12, alpha=0.45, edgecolor="none")
    lim = max(float(y.max()), float(np.max(preds[best_model]))) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("true throughput (Mbps)")
    ax.set_ylabel("predicted (Mbps)")
    ax.set_title(f"{best_model}: predicted vs true")
    fig.tight_layout()
    fig.savefig(out_dir / "scatter.pdf")
    plt.close(fig)
    return best_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("report"))
    parser.add_argument("--repeats", type=int, default=5,
                        help="number of independent group splits to average over")
    parser.add_argument("--skip-ranker", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = impute_features(load_dataset(args.dataset))
    feats = feature_columns(df)
    assert_no_leakage(feats)

    print(f"rows={len(df)} groups={df.group_id.nunique()} features={len(feats)}")
    print(f"overall oracle: {json.dumps(oracle_ceiling(df))}\n")

    all_res, all_chosen = [], {}
    last = None
    for i in range(args.repeats):
        print(f"--- split seed {i} ---")
        res, chosen, test, test_pred = run_once(df, feats, i, args.skip_ranker)
        print(res[["model", "top1_accuracy", "mean_regret_mbps", "mean_spearman"]]
              .to_string(index=False))
        all_res.append(res)
        all_chosen[f"seed_{i}"] = chosen
        last = (test, test_pred)

    res_all = pd.concat(all_res, ignore_index=True)
    agg = res_all.groupby("model")[METRICS].agg(["mean", "std"])

    print("\n=== TEST, averaged over "
          f"{args.repeats} group splits (mean +/- std) ===")
    show = pd.DataFrame({
        "top1": agg[("top1_accuracy", "mean")].map("{:.3f}".format) + " +/- "
                + agg[("top1_accuracy", "std")].map("{:.3f}".format),
        "regret_mbps": agg[("mean_regret_mbps", "mean")].map("{:.3f}".format) + " +/- "
                       + agg[("mean_regret_mbps", "std")].map("{:.3f}".format),
        "regret_frac": agg[("mean_regret_frac", "mean")].map("{:.3f}".format),
        "spearman": agg[("mean_spearman", "mean")].map("{:.3f}".format),
        "r2": agg[("r2", "mean")].map(lambda v: "-" if pd.isna(v) else f"{v:.3f}"),
    }).sort_values("regret_mbps")
    print(show.to_string())

    best_model = make_figures(*last, agg, args.out_dir)
    res_all.to_csv(args.out_dir / "results_raw.csv", index=False)
    agg.to_csv(args.out_dir / "results.csv")
    (args.out_dir / "chosen_hparams.json").write_text(json.dumps(all_chosen, indent=2))

    tex = show.reset_index().rename(columns={
        "model": "Model", "top1": "Top-1", "regret_mbps": "Regret (Mbps)",
        "regret_frac": "Regret (frac)", "spearman": "Spearman", "r2": "$R^2$"})
    latex = tex.to_latex(
        index=False, escape=False,
        caption=(f"AP-selection performance on held-out test groups, averaged over "
                 f"{args.repeats} independent group-wise splits. Regret is the "
                 "throughput given up relative to an oracle that always picks the "
                 "best available AP. $R^2$ is reported only for models that predict "
                 "throughput directly."),
        label="tab:results")
    (args.out_dir / "results_table.tex").write_text(latex)

    (args.out_dir / "summary.json").write_text(json.dumps({
        "rows": int(len(df)), "groups": int(df.group_id.nunique()),
        "n_features": len(feats), "repeats": args.repeats,
        "oracle": oracle_ceiling(df), "best_model": best_model,
    }, indent=2))
    print(f"\nwrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
