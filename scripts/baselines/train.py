#!/usr/bin/env python3
"""This file trains three baseline models on the aggregate table constructed in
./build_datatable.py -- a regularised linear model and two tree ensembles.

Common Heuristics -- strongest signal, least busy channel, and signal traded off
against channel occupancy are also evaluated on their ability to pick the best AP.

The file can be run at several --train-fraction values for a learning curve:
each point fits on that share of the training topologies, keeping validation
and test the same. If regret is still falling at 1.0, then it means
that more simulation data would likely help.

Run:
  .venv/bin/python3 scripts/baselines/train.py data/aggregate.csv \
      --out-dir results/table
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.common.splits import (N_APS_COL, N_FOLDS, TOPOLOGY_COL,  # noqa: E402
                                   split_by_topology)
from scripts.common.evaluate import (LABEL_COL, REGRET_CDF_MARGINS,  # noqa: E402
                                     SCAN_COL, all_metrics, label_spread,
                                     selection_metrics,
                                     validation_selection_key)


def run_fold(df, feats, n_test_fold, n_folds, train_fraction=1.0):
    """Train and Eval with the testing fold fixed at test_fold. Call in a loop
    for K fold cross validation.

    The split is over topologies, so no deployment is both fitted on and tested
    on, and the fold index picks the same partition here as it does for the
    neural models trained in tf/train.py.
    """
    # 1. Hold out topologies, then keep the share of the training ones asked for.
    train, val, test = split_by_topology(df, test_fold=n_test_fold, n_folds=n_folds)
    train = subset_topologies(train, train_fraction, n_test_fold)
    print(f"    training on {train.topology_id.nunique()} topologies, "
          f"{train.scan_id.nunique()} scans")

    # 2. Fit every candidate of all three families; each family keeps the one
    #    setting that decides best on the validation topologies.
    val_pred, test_pred, chosen = fit_models(train, val, test, feats, n_test_fold)

    # 3. Score each family's kept setting, and the heuristics, on the held-out
    #    scans. A rule is scored on selection alone: it ranks APs and does not
    #    estimate Mbps, so its regression cells stay empty.
    rules = heuristic_predictions(test, fit_frame=train)
    res = pd.DataFrame([{"model": name, **all_metrics(test, pred)}
                        for name, pred in test_pred.items()]
                       + [{"model": name, **selection_metrics(test, pred)}
                          for name, pred in rules.items()])
    res["fold"] = n_test_fold

    # 4. Keep the test rows themselves, so regret can be sliced afterwards.
    rows = prediction_rows(test, {**rules, **test_pred}, n_test_fold)
    return res, chosen, test, test_pred, rows


# ---------------------------------------------------------------------------
# 1. Subsets of Train
# ---------------------------------------------------------------------------

# split_by_topology hands back the whole training part. To capture the value of the 
# dataset size, we can train on increasing percentages of the train dataset and 
# see how much it helps. However, when such partitioning is done, care must be 
# taken to keep the distribution of nAPs roughly same across values of fraction.

def subset_topologies(train: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Nested, AP-count-stratified subset of the training topologies.

    Every scan of a chosen topology comes along with it. The permutation
    depends only on `seed`, so a larger fraction with the same seed returns a
    superset, and the points of a learning curve compare like with like.
    """
    if fraction >= 1.0:
        return train
    topology = train.groupby(TOPOLOGY_COL, sort=True).first().reset_index()
    rng = np.random.default_rng(seed)
    selected = []
    for _, stratum in topology.groupby(N_APS_COL, sort=True):
        members = stratum[TOPOLOGY_COL].to_numpy()
        members = members[rng.permutation(len(members))]
        selected.extend(members[:max(1, round(len(members) * fraction))])
    return train[train[TOPOLOGY_COL].isin(selected)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Fitting the three families, each chosen on validation
# ---------------------------------------------------------------------------

# Each family is fitted over a grid, twice: once against throughput and once
# against its log1p. That doubles a family's candidates, and the validation
# topologies then pick one per family - so a family is reported at its own best
# setting rather than at a setting guessed once for all three.


def fit_models(train, val, test, feats, n_test_fold):
    """Each family's winner, as its validation and test predictions, and the
    settings it won with."""
    # imported here rather than at module scope: only fitting needs sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X_tr, y_tr = to_xy(train, feats)
    X_va, _ = to_xy(val, feats)
    X_te, _ = to_xy(test, feats)

    grids = {
        "ridge_linear": (lambda **kw: make_pipeline(StandardScaler(), Ridge(**kw)),
                         [dict(alpha=a) for a in (0.1, 1.0, 10.0, 100.0)]),
        "hist_gbr": (HistGradientBoostingRegressor, [
            dict(max_iter=n, learning_rate=lr, min_samples_leaf=leaf,
                 l2_regularization=1.0, early_stopping=False, random_state=n_test_fold)
            for n in (200, 400) for lr in (0.03, 0.06, 0.1) for leaf in (4, 8, 16)
        ]),
        "random_forest": (RandomForestRegressor, [
            dict(n_estimators=400, min_samples_leaf=leaf, max_features=mf,
                 n_jobs=-1, random_state=n_test_fold)
            for leaf in (1, 2, 4) for mf in ("sqrt", 0.5, 1.0)
        ]),
    }

    val_pred, test_pred, chosen = {}, {}, {}
    for name, (ctor, grid) in grids.items():
        def fit(cfg, is_log_target):
            estimator = ctor(**cfg)
            estimator.fit(X_tr, np.log1p(y_tr) if is_log_target else y_tr)
            return estimator

        def predict(estimator, X, is_log_target):
            p = estimator.predict(X)
            return np.expm1(np.clip(p, -5, 12)) if is_log_target else p

        # Only the validation predictions are kept while searching: holding
        # every candidate's fitted estimator would hold the whole grid in memory.
        cands = []
        for is_log_target in (False, True):
            for cfg in grid:
                cands.append(((cfg, is_log_target),
                              predict(fit(cfg, is_log_target), X_va, is_log_target)))
        (regret, _), (cfg, is_log_target) = choose_by_validation_regret(cands, val)

        # So the winner is fitted a second time, now to predict the test rows too.
        m = fit(cfg, is_log_target)
        for X, store in ((X_va, val_pred), (X_te, test_pred)):
            store[name] = predict(m, X, is_log_target)
        chosen[name] = {k: v for k, v in cfg.items() if k not in ("random_state", "n_jobs")}
        chosen[name].update(is_log_target=is_log_target,
                            val_topology_regret=round(regret, 4))
    return val_pred, test_pred, chosen


def to_xy(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[LABEL_COL].to_numpy(dtype=np.float32)
    return X, y


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


# ---------------------------------------------------------------------------
# 3. The heuristics a client could use instead of a model
# ---------------------------------------------------------------------------

# None of these learns from the corpus, so they are a property of the dataset
# rather than of a training run. Only rssi_minus_busy has anything to fit, and
# it fits on the fold's training topologies like everything else here.

# The occupancy the busy rules read. CCA counts energy that prevented reception
# even where no header was decoded, which is what a chipset's channel-busy
# counter reports.
BUSY_COL = "feat_chan_cca_busy_frac"

# Candidate values of k for the rssi_minus_busy rule, in dB of signal traded
# against a fully busy channel; k = 0 is strongest_rssi. A fitted k on either
# end means the optimum lies outside the range and the range needs widening.
RSSI_BUSY_K_GRID = tuple(float(k) for k in range(0, 81, 5))


def heuristic_predictions(df: pd.DataFrame, fit_frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-AP scores from each rule, for the rows of `df`.

    `fit_frame` supplies the training rows that rssi_minus_busy fits its one
    parameter on; it must not overlap `df`, or that rule is being scored on data
    it was tuned on. Every rule whose input columns are absent from `df` is left
    out of the returned dict rather than faked.
    """
    preds = {}

    # A flat score, so every AP in a scan ties. selection_metrics averages
    # credit and regret across a tie, which makes this the exact expectation
    # for a uniform pick rather than one noisy draw from it.
    preds["random"] = np.zeros(len(df))

    if "feat_ap_rssi_mean" in df:
        preds["strongest_rssi"] = df["feat_ap_rssi_mean"].to_numpy()
    if BUSY_COL in df:
        preds["least_busy_channel"] = -df[BUSY_COL].to_numpy()
    if "feat_ap_rssi_mean" in df and BUSY_COL in df:
        k = fit_rssi_busy_k(fit_frame, BUSY_COL)
        preds["rssi_minus_busy"] = (preds["strongest_rssi"]
                                    - k * df[BUSY_COL].to_numpy())
    return preds


def fit_rssi_busy_k(train_df: pd.DataFrame, busy_column: str,
                    grid: tuple[float, ...] = RSSI_BUSY_K_GRID) -> float:
    """The k of the `RSSI - k*busy` rule with the lowest regret on TRAINING
    rows, so the rule is never scored on rows it was tuned on.

    Ties go to the smaller k. Raises rather than falling back to a fixed k: a
    fitted rule in one experiment and an unfitted one in another would make
    their numbers incomparable.
    """
    if len(train_df) == 0:
        raise ValueError("cannot fit the rssi_minus_busy rule on an empty frame")
    for column in ("feat_ap_rssi_mean", busy_column, LABEL_COL, SCAN_COL):
        if column not in train_df:
            raise ValueError(f"fit frame is missing {column!r}, needed to fit "
                             "the rssi_minus_busy rule")
    r = train_df["feat_ap_rssi_mean"].to_numpy()
    b = train_df[busy_column].to_numpy()
    scores = [(selection_metrics(train_df, r - k * b)["mean_regret_mbps"], k)
              for k in grid]
    return min(scores)[1]


# ---------------------------------------------------------------------------
# 4. The test rows a fold leaves behind
# ---------------------------------------------------------------------------

# A fold's metrics are one number each, which cannot say where a model earns its
# lead. So every test row is kept with what each model predicted for it, and the
# reporting section regroups those rows to answer that. Over a full rotation they
# are the whole corpus, each scan predicted by models that never trained on it.


def prediction_rows(test, predictions, fold):
    """The fold's test rows: the identifiers and labels, the columns the slices
    are cut on, and one pred_* column per model and heuristic."""
    keep = ["topology_id", "scan_id", "ap_index", "label_throughput_mbps",
            "gt_n_aps", "gt_n_hotspots", "feat_ap_rssi_mean"]
    rows = test[[c for c in keep if c in test.columns]].copy()
    rows["fold"] = fold
    for name, p in predictions.items():
        rows[f"pred_{name}"] = p

    # A scan is fully discovered when it heard a beacon from every AP the
    # deployment placed; where it heard fewer, some AP was never discovered.
    group_sizes = rows.groupby("scan_id")["scan_id"].transform("size")
    rows["discovery"] = np.where(group_sizes == rows["gt_n_aps"], "full", "partial")
    return rows


# ---------------------------------------------------------------------------
# Reading the aggregate table
# ---------------------------------------------------------------------------

# main() reads build_datatable.py's table once, checks it and fills its gaps, and
# every fold then draws from that one frame. Nothing here depends on the fold, so
# it all runs once, before the rotation starts.

AP_COL = "ap_index"


def load_dataset(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = {LABEL_COL, SCAN_COL, AP_COL} - set(df.columns)
    if missing:
        raise ValueError(f"dataset missing required columns: {missing}")
    return df


def impute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy in which no feat_* column contains NaN, filling with 0.0.

    cache_dataset.py writes MISSING_RSSI_DBM wherever a level was not heard, so
    the columns that reach here missing are counts and rates, and zero is their
    reading of absent. Callers that skip this step and hand a NaN to a model get
    NaN scores and a metric that refuses them.
    """
    df = df.copy()
    for c in feature_columns(df):
        df[c] = df[c].fillna(0.0)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("feat_")]


# build_datatable.py emits one row per (scan, discovered AP). Only the feat_* columns
# of that row are observable before joining; the label, the identifiers and the
# gt_* deployment sizes exist to score and slice, and never reach a model.

def assert_no_leakage(feature_cols: list[str]) -> None:
    bad = [c for c in feature_cols if not c.startswith("feat_")]
    if bad:
        raise AssertionError(f"non-observable columns used as features: {bad}")


# ---------------------------------------------------------------------------
# Reporting: the tables and the two figures
# ---------------------------------------------------------------------------

# Each writer below prints its table and leaves the same numbers on disk, as CSV
# to read again and as LaTeX for the report. The slice report regroups the
# per-row predictions that step 4 kept.


def summary_table(agg: pd.DataFrame) -> pd.DataFrame:
    """One row per model, formatted for display, lowest mean regret first."""
    show = pd.DataFrame({
        "mae": agg[("mae", "mean")].map(lambda v: "-" if pd.isna(v) else f"{v:.3f}"),
        "top1": agg[("top1_rate", "mean")].map("{:.3f}".format),
        "regret_mbps": agg[("mean_regret_mbps", "mean")].map("{:.3f}".format) + " +/- "
                       + agg[("mean_regret_mbps", "std")].map("{:.3f}".format),
        **{f"regret_cdf_{round(m * 100)}pct":
           agg[(f"regret_cdf_{round(m * 100)}pct", "mean")].map("{:.3f}".format)
           for m in REGRET_CDF_MARGINS},
        "regret_frac": agg[("mean_regret_frac", "mean")].map("{:.3f}".format),
        "spearman": agg[("mean_spearman", "mean")].map("{:.3f}".format),
        "r2": agg[("r2", "mean")].map(lambda v: "-" if pd.isna(v) else f"{v:.3f}"),
        "r2_log": agg[("r2_log", "mean")].map(lambda v: "-" if pd.isna(v) else f"{v:.3f}"),
    })
    return show.loc[agg[("mean_regret_mbps", "mean")].sort_values().index]


def write_results_table(show: pd.DataFrame, out_dir: Path, n_folds: int) -> None:
    """The headline table, as LaTeX for the report."""
    tex = show.reset_index()
    tex["model"] = tex["model"].str.replace("_", r"\_", regex=False)
    tex = tex.rename(columns={
        "model": "Model", "mae": "MAE (Mbps)", "top1": "Top-1",
        "regret_mbps": "Regret (Mbps)",
        **{f"regret_cdf_{round(m * 100)}pct": f"Regret CDF ({round(m * 100)}\\%)"
           for m in REGRET_CDF_MARGINS},
        "regret_frac": "Regret (frac)",
        "spearman": "Spearman",
        "r2": "$R^2$", "r2_log": "$R^2_{\\log}$"})
    (out_dir / "results_table.tex").write_text(tex.to_latex(
        index=False, escape=False,
        caption=(f"AP-selection performance on held-out test groups, over the "
                 f"{n_folds} folds of a rotation, each tested on topologies no "
                 "other part of its run saw. Regret is the "
                 "throughput given up relative to an oracle that always picks the "
                 "best available AP. $R^2$ is reported only for models that predict "
                 "throughput directly."),
        label="tab:results"))


def write_stratified(preds_df: pd.DataFrame, out_dir: Path) -> None:
    """Regret broken down by dataset dimension."""
    # Every prediction column is worth a breakdown except random, whose regret
    # carries the variance of the draw rather than anything about the slice.
    models = [column[5:] for column in preds_df.columns
              if column.startswith("pred_") and column != "pred_random"]
    dimensions = dimension_report(preds_df, models)
    print("\n=== MEAN REGRET BY DATASET DIMENSION (Mbps, pooled over folds) ===")
    print(dimensions.to_string(index=False))
    dimensions.to_csv(out_dir / "stratified_dimensions.csv", index=False)


def dimension_report(preds_df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Regret by simulator regime and full/partial passive discovery."""
    rows = []
    for dimension in ("gt_n_aps", "gt_n_hotspots", "discovery"):
        if dimension not in preds_df:
            continue
        for value, subset in preds_df.groupby(dimension):
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


def make_figures(test, preds, agg, best_model: str, out_dir: Path):
    """Two figures: every model's regret side by side, and the best one's
    predictions against the truth."""
    # imported here rather than at module scope: only the figures need matplotlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Regret, best first, with the learned models coloured apart from the
    # heuristics they have to beat; the bar's error is the spread over folds.
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

    # The scatter shows one fold's rows, but `best_model` is chosen by the
    # aggregate over every fold, so the plot does not follow whichever model
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


# ---------------------------------------------------------------------------
# The command line, and running every fold
# ---------------------------------------------------------------------------

# main() runs every fold of the rotation, stacks their per-fold metrics into one
# frame and averages it. What that leaves on disk is the per-fold numbers, their
# fold-average, the settings each family won with, the sliced regret and the figures.


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("results/main"))
    parser.add_argument("--folds", type=int, default=N_FOLDS,
                        help="how many folds the topologies are dealt into; every "
                             "fold is run, so each topology is tested exactly once")
    parser.add_argument("--train-fraction", type=float, default=1.0,
                        help="share of each fold's training topologies to fit on; "
                             "run the same command at several fractions for a "
                             "learning curve, and the validation and test "
                             "partitions stay as they are")
    args = parser.parse_args()
    if not 0 < args.train_fraction <= 1:
        parser.error("--train-fraction must lie in (0, 1]")
    return args


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # The table is read, imputed and checked once, then every fold draws from it.
    df = impute_features(load_dataset(args.dataset))
    feats = feature_columns(df)
    assert_no_leakage(feats)
    print(f"rows={len(df)} scans={df.scan_id.nunique()} features={len(feats)}")
    print(f"overall oracle: {json.dumps(label_spread(df))}\n")

    all_res, all_chosen, pred_rows = [], {}, []
    # The scatter figure shows the rows of one fold; the last is as good as any,
    # and keeping only it avoids holding every fold's frame in memory.
    final_test, final_pred = None, None
    for fold in range(args.folds):
        print(f"--- fold {fold} ---")
        res, chosen, test, test_pred, rows = run_fold(
            df, feats, fold, args.folds, args.train_fraction)
        pred_rows.append(rows)
        print(res[["model", "mae", "r2", "mean_regret_mbps"]]
              .to_string(index=False))
        all_res.append(res)
        all_chosen[f"fold_{fold}"] = chosen
        final_test, final_pred = test, test_pred

    # Averaging the folds is what makes a difference between two models
    # readable: the spread over folds says how much of one is the luck of which
    # topologies a fold happened to hold.
    # The learned models are whatever run_fold actually trained. Naming them by
    # excluding the heuristics instead would silently promote a newly added
    # heuristic into the choice of best model.
    learned = sorted(final_pred)
    res_all = pd.concat(all_res, ignore_index=True)
    agg = res_all.drop(columns="fold").groupby("model", sort=False).agg(["mean", "std"])
    show = summary_table(agg)
    print(f"\n=== TEST, averaged over {args.folds} folds (mean +/- std) ===")
    print(show.to_string())

    preds_df = pd.concat(pred_rows, ignore_index=True)
    preds_df.to_csv(args.out_dir / "test_predictions.csv", index=False)
    write_stratified(preds_df, args.out_dir)
    write_results_table(show, args.out_dir, args.folds)

    # Chosen on the aggregate over every fold, so the figure does not follow
    # whichever model happened to win the last one.
    best_model = min(learned, key=lambda name: agg.loc[name, ("mean_regret_mbps", "mean")])
    make_figures(final_test, final_pred, agg, best_model, args.out_dir)

    # Everything a later question might need: the per-fold metrics, their
    # average, the settings each family won with, and the run's own shape.
    res_all.to_csv(args.out_dir / "results_raw.csv", index=False)
    agg.to_csv(args.out_dir / "results.csv")
    (args.out_dir / "chosen_hparams.json").write_text(json.dumps(all_chosen, indent=2))
    (args.out_dir / "summary.json").write_text(json.dumps({
        "rows": int(len(df)), "groups": int(df.scan_id.nunique()),
        "topologies": int(df.topology_id.nunique()) if "topology_id" in df else None,
        "n_features": len(feats), "folds": args.folds,
        "oracle": label_spread(df), "best_model": best_model,
    }, indent=2))
    print(f"\nwrote results to {args.out_dir}")



if __name__ == "__main__":
    main()