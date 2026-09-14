#!/usr/bin/env python3
"""Which model picks the best access point from the summary feature table?

This is the headline experiment on the tabular view of the data. Each row is
one access point a client could have joined, described by summary statistics of
what the client heard before joining - signal strength, how busy the channel
was, how this option compares to the others on offer - and labelled with the
throughput it would actually have delivered. Rows sharing a scan_id are the
options in one scan, and the job is to score them so the best comes top.

Three learned models are trained on that table: a regularised linear model and
two tree ensembles, each predicting throughput per row. All are reported against
the heuristics a real client could run instead - strongest signal, least busy
channel, and signal traded off against channel occupancy. A model that cannot
beat strongest signal has not justified itself.

The protocol is strict because the corpus is small enough that sloppiness would
dominate the result:

  * splits are by topology, so repeated observations of one deployment never
    straddle a split. They correlate almost perfectly, and a leak here would
    quietly inflate every number below.
  * hyperparameters are chosen on validation only, never on test.
  * everything is repeated over several independent splits and reported as
    mean +/- std, because one split of a few hundred scans is noisy
    enough that a single number would mislead.

Beyond the headline table it also reports where the learning actually pays:
scans are split by whether strongest-signal already picks the winner, and
sliced by deployment size, hotspot count, client placement and whether the scan
found every access point.

Run:
  .venv/bin/python3 scripts/train/train_eval.py data/aggregate.csv \
      --out-dir results/static

TODO: this overlaps scripts/models/baseline.py, which fits the same tree models
from its own entry point, scoring on validation and adding permutation
importance. One of the two is to be deleted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.models.data import (assert_no_leakage, feature_columns, scan_sample_weights,  # noqa: E402
                         impute_features, load_dataset, split_by_topology, to_xy)
from scripts.models.evaluate import (evaluate_all, label_spread,  # noqa: E402
                             selection_metrics, validation_selection_key)

METRICS = ["top1_accuracy", "mean_regret_mbps", "median_regret_mbps",
           "mean_regret_frac", "mean_spearman", "topology_top1_accuracy",
           "topology_mean_regret_mbps", "mae", "rmse", "r2", "r2_log"]

# A scan counts as one where signal strength already gives the right
# answer if the strongest option is within this many Mbps of the best. Matches
# the tie tolerance selection_metrics() applies to top-1.
RSSI_OPTIMAL_TOLERANCE_MBPS = 0.5


def choose_by_validation_regret(candidates, val):
    """Return the (key, payload) of the candidate that decides best on validation.

    `candidates` pairs an arbitrary payload - whatever the caller needs to
    rebuild or look up the winner - with that candidate's validation
    predictions. Ordering is validation_selection_key()'s, so every script in
    this directory selects the same way.
    """
    best = ((np.inf, np.inf), None)
    for payload, pred in candidates:
        key = validation_selection_key(val, pred)
        if key < best[0]:
            best = (key, payload)
    return best


def fit_tree_models(train, val, test, feats, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X_tr, y_tr = to_xy(train, feats)
    X_va, _ = to_xy(val, feats)
    X_te, _ = to_xy(test, feats)
    train_weights = scan_sample_weights(train)

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
        # Ridge is wrapped in a pipeline, so its per-row weights have to be
        # addressed by step name; the bare estimators take them directly.
        weight_arg = ("ridge__sample_weight" if name == "ridge_linear"
                      else "sample_weight")

        def fit(cfg, log_target):
            estimator = ctor(**cfg)
            estimator.fit(X_tr, np.log1p(y_tr) if log_target else y_tr,
                          **{weight_arg: train_weights})
            return estimator

        def predict(estimator, X, log_target):
            p = estimator.predict(X)
            return np.expm1(np.clip(p, -5, 12)) if log_target else p

        cands = []
        for log_target in (False, True):
            for cfg in grid:
                cands.append(((cfg, log_target),
                              predict(fit(cfg, log_target), X_va, log_target)))
        (regret, _), (cfg, log_target) = choose_by_validation_regret(cands, val)

        # Refit the winner: only its validation predictions were kept above.
        m = fit(cfg, log_target)
        for X, store in ((X_va, val_pred), (X_te, test_pred)):
            store[name] = predict(m, X, log_target)
        chosen[name] = {k: v for k, v in cfg.items() if k not in ("random_state", "n_jobs")}
        chosen[name].update(log_target=log_target,
                            val_topology_regret=round(regret, 4))
    return val_pred, test_pred, chosen


def pool_across_splits(preds_df: pd.DataFrame) -> pd.DataFrame:
    """Stack every split's test rows into one frame with distinct group ids.

    The same physical scan can land in the test partition of more than
    one split. Prefixing the split seed keeps those evaluations separate;
    without it a later groupby would concatenate duplicate option rows into a
    single scan that never existed.
    """
    pooled = preds_df.copy()
    pooled["scan_id"] = (pooled["split_seed"].astype(str) + ":" +
                          pooled["scan_id"].astype(str))
    return pooled


def stratified_report(preds_df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Where does learning actually pay?

    Split the test groups by whether the strongest-RSSI rule already picks
    the best AP. In the groups where it does, a model can only match it; all
    the value has to come from the groups where signal strength is
    misleading, and those are worth looking at separately.
    """
    pooled = pool_across_splits(preds_df)
    rows = []
    for label, sub in pooled.groupby("stratum"):
        row = {"stratum": label, "scans": sub.scan_id.nunique()}
        for m in models + ["strongest_rssi"]:
            row[m] = selection_metrics(sub, sub[f"pred_{m}"].to_numpy())["mean_regret_mbps"]
        rows.append(row)
    return pd.DataFrame(rows)


def dimension_report(preds_df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Regret by simulator regime and full/partial passive discovery."""
    pooled = pool_across_splits(preds_df)
    rows = []
    for dimension in ("gt_n_aps", "gt_n_hotspots", "discovery"):
        if dimension not in pooled:
            continue
        for value, subset in pooled.groupby(dimension):
            row = {
                "dimension": dimension,
                "value": value,
                "scans": subset.scan_id.nunique(),
            }
            for model in models:
                row[model] = selection_metrics(
                    subset, subset[f"pred_{model}"].to_numpy())["mean_regret_mbps"]
            rows.append(row)
    return pd.DataFrame(rows)


def run_once(df, feats, seed):
    train, val, test = split_by_topology(df, seed=seed)
    val_pred, test_pred, chosen = fit_tree_models(train, val, test, feats, seed)
    res = evaluate_all(test, test_pred, train)
    res["split_seed"] = seed

    # keep per-row test predictions so results can be sliced afterwards
    from scripts.models.evaluate import baseline_predictions
    keep = ["topology_id", "scan_id", "ap_index", "label_throughput_mbps",
            "gt_n_aps", "gt_n_hotspots", "gt_n_channels", "gt_true_distance", "feat_ap_rssi_mean"]
    rows = test[[c for c in keep if c in test.columns]].copy()
    rows["split_seed"] = seed
    for name, p in {**baseline_predictions(test, fit_frame=train), **test_pred}.items():
        rows[f"pred_{name}"] = p
    group_sizes = rows.groupby("scan_id")["scan_id"].transform("size")
    rows["discovery"] = np.where(group_sizes == rows["gt_n_aps"], "full", "partial")

    # a group is "rssi-optimal" when the strongest AP is already the best one
    rssi_regret = {}
    for gid, g in rows.groupby("scan_id"):
        y = g["label_throughput_mbps"].to_numpy()
        scores = g["pred_strongest_rssi"].to_numpy()
        tied = np.isclose(scores, scores.max(), rtol=1e-12, atol=1e-12)
        chosen_y = float(y[tied].mean())
        rssi_regret[gid] = y.max() - chosen_y
    rows["stratum"] = rows.scan_id.map(
        lambda g: "RSSI already optimal"
        if rssi_regret[g] <= RSSI_OPTIMAL_TOLERANCE_MBPS else "RSSI misleading")
    return res, chosen, test, test_pred, rows


def make_figures(test, preds, agg, best_model: str, out_dir: Path):
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

    # The scatter shows one split's rows, but `best_model` is chosen by the
    # aggregate over every split, so the plot does not follow whichever model
    # happened to win the last one.
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("results/main"))
    parser.add_argument("--repeats", type=int, default=5,
                        help="number of independent topology splits to average over")
    return parser.parse_args()


def summary_table(agg: pd.DataFrame) -> pd.DataFrame:
    """One row per model, formatted for display, lowest mean regret first."""
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
    return show.loc[agg[("mean_regret_mbps", "mean")].sort_values().index]


def write_stratified(preds_df: pd.DataFrame, learned: list[str], out_dir: Path) -> None:
    """Regret broken down by stratum and by dataset dimension."""
    strat = stratified_report(preds_df, learned)
    print("\n=== MEAN REGRET BY STRATUM (Mbps, pooled over splits) ===")
    print(strat.to_string(index=False))
    strat.to_csv(out_dir / "stratified.csv", index=False)
    strat_tex = strat.copy()
    strat_tex.columns = [c.replace('_', r'\_') for c in strat_tex.columns]
    (out_dir / "stratified_table.tex").write_text(strat_tex.to_latex(
        index=False, float_format="%.3f", escape=False,
        caption="Mean regret (Mbps) split by whether the strongest-RSSI rule "
                "already selects the best AP. Pooled across all test splits.",
        label="tab:stratified"))

    models = [column[5:] for column in preds_df.columns
              if column.startswith("pred_") and column != "pred_random"]
    dimensions = dimension_report(preds_df, models)
    print("\n=== MEAN REGRET BY DATASET DIMENSION (Mbps, pooled over splits) ===")
    print(dimensions.to_string(index=False))
    dimensions.to_csv(out_dir / "stratified_dimensions.csv", index=False)


def write_results_table(show: pd.DataFrame, out_dir: Path, repeats: int) -> None:
    """The headline table, as LaTeX for the report."""
    tex = show.reset_index()
    tex["model"] = tex["model"].str.replace("_", r"\_", regex=False)
    tex = tex.rename(columns={
        "model": "Model", "top1": "Top-1", "regret_mbps": "Regret (Mbps)",
        "regret_frac": "Regret (frac)", "topology_regret": "Topology regret",
        "spearman": "Spearman",
        "r2": "$R^2$", "r2_log": "$R^2_{\\log}$"})
    (out_dir / "results_table.tex").write_text(tex.to_latex(
        index=False, escape=False,
        caption=(f"AP-selection performance on held-out test groups, averaged over "
                 f"{repeats} independent group-wise splits. Regret is the "
                 "throughput given up relative to an oracle that always picks the "
                 "best available AP. $R^2$ is reported only for models that predict "
                 "throughput directly."),
        label="tab:results"))


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = impute_features(load_dataset(args.dataset))
    feats = feature_columns(df)
    assert_no_leakage(feats)
    print(f"rows={len(df)} scans={df.scan_id.nunique()} features={len(feats)}")
    print(f"overall oracle: {json.dumps(label_spread(df))}\n")

    all_res, all_chosen, pred_rows = [], {}, []
    # The scatter figure shows the rows of one split; the last is as good as any,
    # and keeping only it avoids holding every split's frame in memory.
    final_test, final_pred = None, None
    for i in range(args.repeats):
        print(f"--- split seed {i} ---")
        res, chosen, test, test_pred, rows = run_once(
            df, feats, i)
        pred_rows.append(rows)
        print(res[["model", "top1_accuracy", "mean_regret_mbps", "mean_spearman"]]
              .to_string(index=False))
        all_res.append(res)
        all_chosen[f"seed_{i}"] = chosen
        final_test, final_pred = test, test_pred

    # The learned models are whatever run_once actually trained. Naming them by
    # excluding the heuristics instead would silently promote a newly added
    # heuristic into the learned-model tables.
    learned = sorted(final_pred)
    res_all = pd.concat(all_res, ignore_index=True)
    agg = res_all.groupby("model")[METRICS].agg(["mean", "std"])
    show = summary_table(agg)
    print(f"\n=== TEST, averaged over {args.repeats} group splits (mean +/- std) ===")
    print(show.to_string())

    preds_df = pd.concat(pred_rows, ignore_index=True)
    preds_df.to_csv(args.out_dir / "test_predictions.csv", index=False)
    write_stratified(preds_df, learned, args.out_dir)
    write_results_table(show, args.out_dir, args.repeats)

    # Chosen on the aggregate over every split, so the figure does not follow
    # whichever model happened to win the last one.
    best_model = min(learned, key=lambda name: agg.loc[name, ("mean_regret_mbps", "mean")])
    make_figures(final_test, final_pred, agg, best_model, args.out_dir)
    res_all.to_csv(args.out_dir / "results_raw.csv", index=False)
    agg.to_csv(args.out_dir / "results.csv")
    (args.out_dir / "chosen_hparams.json").write_text(json.dumps(all_chosen, indent=2))
    (args.out_dir / "summary.json").write_text(json.dumps({
        "rows": int(len(df)), "groups": int(df.scan_id.nunique()),
        "topologies": int(df.topology_id.nunique()) if "topology_id" in df else None,
        "n_features": len(feats), "repeats": args.repeats,
        "oracle": label_spread(df), "best_model": best_model,
    }, indent=2))
    print(f"\nwrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
