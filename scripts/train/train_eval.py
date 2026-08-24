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

Outputs a results table (CSV + LaTeX) and evaluation figures.

Run:  python scripts/train_eval.py data/dataset.csv --out-dir results/main/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.data import (assert_no_leakage, feature_columns, group_sample_weights,  # noqa: E402
                         impute_features, load_dataset, split_by_group, to_xy)
from models.evaluate import evaluate_all, oracle_ceiling, selection_metrics  # noqa: E402

METRICS = ["top1_accuracy", "mean_regret_mbps", "median_regret_mbps",
           "mean_regret_frac", "mean_spearman", "topology_top1_accuracy",
           "topology_mean_regret_mbps", "mae", "rmse", "r2", "r2_log"]


def _select(candidates, val):
    """Pick by validation regret, breaking ties on within-group ordering.

    Regret bottoms out at zero as soon as a configuration gets every
    validation group right, and on a few hundred groups several will. Left
    unbroken, the tie would be settled by grid order rather than evidence.
    """
    best = ((np.inf, np.inf), None)
    for key_extra, pred in candidates:
        sm = selection_metrics(val, pred)
        regret = sm.get("topology_mean_regret_mbps", sm["mean_regret_mbps"])
        key = (regret, -sm["mean_spearman"])
        if key < best[0]:
            best = (key, key_extra)
    return best


def fit_tree_models(train, val, test, feats, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X_tr, y_tr = to_xy(train, feats)
    X_va, _ = to_xy(val, feats)
    X_te, _ = to_xy(test, feats)
    train_weights = group_sample_weights(train)

    grids = {
        # A fitted linear model separates two questions that the tree
        # results conflate: are the features informative at all, and is a
        # nonlinear model actually needed to exploit them?
        "ridge_linear": (lambda **kw: make_pipeline(StandardScaler(), Ridge(**kw)),
                         [dict(alpha=a) for a in (0.1, 1.0, 10.0, 100.0)]),
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
                target = np.log1p(y_tr) if log_target else y_tr
                fit_args = ({"ridge__sample_weight": train_weights}
                            if name == "ridge_linear" else
                            {"sample_weight": train_weights})
                m.fit(X_tr, target, **fit_args)
                p = m.predict(X_va)
                p = np.expm1(np.clip(p, -5, 12)) if log_target else p
                cands.append(((cfg, log_target), p))
        (regret, _), (cfg, log_target) = _select(cands, val)

        m = ctor(**cfg)
        target = np.log1p(y_tr) if log_target else y_tr
        fit_args = ({"ridge__sample_weight": train_weights}
                    if name == "ridge_linear" else
                    {"sample_weight": train_weights})
        m.fit(X_tr, target, **fit_args)
        for X, store in ((X_va, val_pred), (X_te, test_pred)):
            p = m.predict(X)
            store[name] = np.expm1(np.clip(p, -5, 12)) if log_target else p
        chosen[name] = {k: v for k, v in cfg.items() if k not in ("random_state", "n_jobs")}
        chosen[name].update(log_target=log_target,
                            val_topology_regret=round(regret, 4))
    return val_pred, test_pred, chosen


def stratified_report(preds_df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Where does learning actually pay?

    Split the test groups by whether the strongest-RSSI rule already picks
    the best AP. In the groups where it does, a model can only match it; all
    the value has to come from the groups where signal strength is
    misleading, and those are worth looking at separately.
    """
    # The same physical choice set may enter the test partition under more
    # than one repeated split. Keep those evaluations separate; grouping only
    # by group_id would concatenate duplicate option rows into one fake set.
    pooled = preds_df.copy()
    pooled["group_id"] = (pooled["split_seed"].astype(str) + ":" +
                          pooled["group_id"].astype(str))
    rows = []
    for label, sub in pooled.groupby("stratum"):
        row = {"stratum": label, "groups": sub.group_id.nunique()}
        for m in models + ["strongest_rssi"]:
            row[m] = selection_metrics(sub, sub[f"pred_{m}"].to_numpy())["mean_regret_mbps"]
        rows.append(row)
    return pd.DataFrame(rows)


def dimension_report(preds_df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Regret by simulator regime and full/partial passive discovery."""
    pooled = preds_df.copy()
    pooled["group_id"] = (pooled["split_seed"].astype(str) + ":" +
                          pooled["group_id"].astype(str))
    rows = []
    for dimension in ("gt_n_aps", "gt_n_hotspots", "gt_candidate_stratum",
                      "discovery"):
        if dimension not in pooled:
            continue
        for value, subset in pooled.groupby(dimension):
            row = {
                "dimension": dimension,
                "value": value,
                "groups": subset.group_id.nunique(),
            }
            for model in models:
                row[model] = selection_metrics(
                    subset, subset[f"pred_{model}"].to_numpy())["mean_regret_mbps"]
            rows.append(row)
    return pd.DataFrame(rows)


def fit_ranker(train, val, test, feats, seed, epochs, patience):
    import torch

    from models.ranker import (SetRanker, Standardizer, make_groups, predict_rows,
                               train as train_ranker)

    scaler = Standardizer(train[feats].to_numpy(dtype=np.float32))
    tr, va, te = (make_groups(d, feats, scaler) for d in (train, val, test))

    val_pred, test_pred, chosen = {}, {}, {}
    # "mlp_pointwise" is the same network with the set context removed - the
    # ablation that says whether seeing the alternatives is worth anything,
    # as opposed to the extra depth and parameters that come with it.
    for name, use_context in (("set_ranker", True), ("mlp_pointwise", False)):
        cands, models = [], {}
        # alpha stays below 1 so the pointwise term is always present: a
        # purely ranking-based score is defined only up to a monotone transform,
        # which leaves ranking intact but makes regression metrics meaningless.
        # Five configurations cover calibration-heavy, balanced, and
        # ranking-heavy objectives while varying the regret scale around the
        # balanced setting. The old 4 x 3 Cartesian grid mostly repeated the
        # same saturated pair weights and multiplied training cost by 2.4.
        for alpha, temp in ((0.3, 5.0), (0.7, 2.0), (0.7, 5.0),
                            (0.7, 10.0), (0.9, 5.0)):
            torch.manual_seed(seed)
            m = SetRanker(len(feats), dropout=0.1, use_context=use_context)
            m = train_ranker(m, tr, va, epochs=epochs, lr=3e-3,
                             weight_decay=1e-3, alpha=alpha, temperature=temp,
                             seed=seed, verbose=False, patience=patience)
            p = np.expm1(np.clip(predict_rows(m, va, len(val)), -5, 12))
            cands.append(((alpha, temp), p))
            models[(alpha, temp)] = m
        (regret, _), (alpha, temp) = _select(cands, val)
        model = models[(alpha, temp)]
        val_pred[name] = np.expm1(np.clip(predict_rows(model, va, len(val)), -5, 12))
        test_pred[name] = np.expm1(np.clip(predict_rows(model, te, len(test)), -5, 12))
        chosen[name] = {"alpha": alpha, "temperature": temp,
                        "val_topology_regret": round(regret, 4)}
    return val_pred, test_pred, chosen


def run_once(df, feats, seed, skip_ranker, ranker_epochs, ranker_patience):
    train, val, test = split_by_group(df, seed=seed)
    val_pred, test_pred, chosen = fit_tree_models(train, val, test, feats, seed)
    if not skip_ranker:
        v, t, c = fit_ranker(train, val, test, feats, seed,
                             ranker_epochs, ranker_patience)
        val_pred.update(v)
        test_pred.update(t)
        chosen.update(c)
    res = evaluate_all(test, test_pred)
    res["split_seed"] = seed

    # keep per-row test predictions so results can be sliced afterwards
    from models.evaluate import baseline_predictions
    keep = ["topology_id", "group_id", "ap_index", "label_throughput_mbps",
            "gt_n_aps", "gt_n_hotspots", "gt_candidate_stratum",
            "gt_n_channels", "gt_true_distance", "feat_ap_rssi_mean"]
    rows = test[[c for c in keep if c in test.columns]].copy()
    rows["split_seed"] = seed
    for name, p in {**baseline_predictions(test), **test_pred}.items():
        rows[f"pred_{name}"] = p
    group_sizes = rows.groupby("group_id")["group_id"].transform("size")
    rows["discovery"] = np.where(group_sizes == rows["gt_n_aps"], "full", "partial")

    # a group is "rssi-optimal" when the strongest AP is already the best one
    rssi_regret = {}
    for gid, g in rows.groupby("group_id"):
        y = g["label_throughput_mbps"].to_numpy()
        scores = g["pred_strongest_rssi"].to_numpy()
        tied = np.isclose(scores, scores.max(), rtol=1e-12, atol=1e-12)
        chosen_y = float(y[tied].mean())
        rssi_regret[gid] = y.max() - chosen_y
    rows["stratum"] = rows.group_id.map(
        lambda g: "RSSI already optimal" if rssi_regret[g] <= 0.5 else "RSSI misleading")
    return res, chosen, test, test_pred, rows


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

    # Pick the learned model by aggregate test reporting, not whichever model
    # happened to win the final repeated split used for this scatter plot.
    best_model = min(preds, key=lambda name: agg.loc[name, ("mean_regret_mbps", "mean")])
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
    parser.add_argument("--out-dir", type=Path, default=Path("results/main"))
    parser.add_argument("--repeats", type=int, default=5,
                        help="number of independent group splits to average over")
    parser.add_argument("--skip-ranker", action="store_true")
    parser.add_argument("--ranker-epochs", type=int, default=200)
    parser.add_argument("--ranker-patience", type=int, default=25)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = impute_features(load_dataset(args.dataset))
    feats = feature_columns(df)
    assert_no_leakage(feats)

    print(f"rows={len(df)} groups={df.group_id.nunique()} features={len(feats)}")
    print(f"overall oracle: {json.dumps(oracle_ceiling(df))}\n")

    all_res, all_chosen, pred_rows = [], {}, []
    last = None
    for i in range(args.repeats):
        print(f"--- split seed {i} ---")
        res, chosen, test, test_pred, rows = run_once(
            df, feats, i, args.skip_ranker, args.ranker_epochs,
            args.ranker_patience)
        pred_rows.append(rows)
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
        "topology_regret": agg[("topology_mean_regret_mbps", "mean")].map("{:.3f}".format),
        "spearman": agg[("mean_spearman", "mean")].map("{:.3f}".format),
        "r2": agg[("r2", "mean")].map(lambda v: "-" if pd.isna(v) else f"{v:.3f}"),
        "r2_log": agg[("r2_log", "mean")].map(lambda v: "-" if pd.isna(v) else f"{v:.3f}"),
    })
    show = show.loc[agg[("mean_regret_mbps", "mean")].sort_values().index]
    print(show.to_string())

    preds_df = pd.concat(pred_rows, ignore_index=True)
    preds_df.to_csv(args.out_dir / "test_predictions.csv", index=False)
    learned = [c[5:] for c in preds_df.columns if c.startswith("pred_")
               and c[5:] not in ("random", "strongest_rssi", "least_busy_channel",
                                 "rssi_minus_busy")]
    strat = stratified_report(preds_df, learned)
    print("\n=== MEAN REGRET BY STRATUM (Mbps, pooled over splits) ===")
    print(strat.to_string(index=False))
    strat.to_csv(args.out_dir / "stratified.csv", index=False)
    with open(args.out_dir / "stratified_table.tex", "w") as fh:
        strat_tex = strat.copy()
        strat_tex.columns = [c.replace('_', r'\_') for c in strat_tex.columns]
        fh.write(strat_tex.to_latex(index=False, float_format="%.3f", escape=False,
            caption="Mean regret (Mbps) split by whether the strongest-RSSI rule "
                    "already selects the best AP. Pooled across all test splits.",
            label="tab:stratified"))

    dimension_models = [column[5:] for column in preds_df.columns
                        if column.startswith("pred_") and column != "pred_random"]
    dimensions = dimension_report(preds_df, dimension_models)
    print("\n=== MEAN REGRET BY DATASET DIMENSION (Mbps, pooled over splits) ===")
    print(dimensions.to_string(index=False))
    dimensions.to_csv(args.out_dir / "stratified_dimensions.csv", index=False)

    best_model = make_figures(*last, agg, args.out_dir)
    res_all.to_csv(args.out_dir / "results_raw.csv", index=False)
    agg.to_csv(args.out_dir / "results.csv")
    (args.out_dir / "chosen_hparams.json").write_text(json.dumps(all_chosen, indent=2))

    tex = show.reset_index()
    tex["model"] = tex["model"].str.replace("_", r"\_", regex=False)
    tex = tex.rename(columns={
        "model": "Model", "top1": "Top-1", "regret_mbps": "Regret (Mbps)",
        "regret_frac": "Regret (frac)", "topology_regret": "Topology regret",
        "spearman": "Spearman",
        "r2": "$R^2$", "r2_log": "$R^2_{\\log}$"})
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
        "topologies": int(df.topology_id.nunique()) if "topology_id" in df else None,
        "n_features": len(feats), "repeats": args.repeats,
        "oracle": oracle_ceiling(df), "best_model": best_model,
    }, indent=2))
    print(f"\nwrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
