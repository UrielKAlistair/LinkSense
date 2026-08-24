#!/usr/bin/env python3
"""Train temporal throughput-regression and AP-ranking transformers.

Both models use the same architecture and ordered passive-scan input. The
regressor predicts log1p throughput for every AP. The ranker instead learns a
regret-aware probability distribution over the options, without pretending
that its scores are calibrated Mbps estimates.

Run:
  python scripts/train_temporal.py data/pilot_temporal.npz \
      --out-dir results/pilot_temporal
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.evaluate import (baseline_predictions, regression_metrics,  # noqa: E402
                             random_selection_metrics, selection_metrics)
from models.temporal import (MaskedStandardizer, TemporalCorpus,  # noqa: E402
                             TemporalSetTransformer)


def corpus_observables(corpus: TemporalCorpus) -> dict[str, np.ndarray]:
    """Per-option signal and occupancy read off THIS corpus's own observation.

    The heuristic baselines have to see what the model saw. Taking them from
    corpus.static instead reads the feat_* table in the CSV, which is always
    the single-pass sweep whatever corpus is being scored - so a continuous
    model was being compared against a 110 ms strongest-RSSI rule, and the
    baseline rows came out byte-identical across every experiment.

    RSSI averages only the bins that carried a real reading; occupancy averages
    every valid bin, because an idle bin is a genuine zero.
    """
    names = list(corpus.temporal_features)

    def column(*candidates):
        for name in candidates:
            if name in names:
                return names.index(name)
        return None

    rssi_i = column("option_rssi_mean")
    seen_i = column("option_observed")
    busy_i = column("cca_busy_fraction", "own_cca_busy_fraction")
    # In the sweep corpus cca_busy_fraction describes the channel the radio is
    # WATCHING, which is the same for every option at a given step. Averaging it
    # unrestricted gives every option in a group the same number, the baseline
    # ties on all of them, and least_busy_channel silently degenerates into
    # random choice. Restrict it to the steps that were actually on this
    # option's channel. The continuous corpus needs no such restriction: its
    # own_cca_busy_fraction is already per option.
    own_i = column("is_option_channel")
    valid = corpus.time_mask[:, None, :]

    out = {}
    if rssi_i is not None:
        rssi = corpus.temporal[:, :, :, rssi_i]
        live = valid & (corpus.temporal[:, :, :, seen_i] > 0.5) if seen_i is not None else valid
        total = np.where(live, rssi, 0.0).sum(axis=2)
        count = live.sum(axis=2)
        # an option never heard live keeps its filled level rather than a zero
        fallback = np.where(valid, rssi, np.nan)
        out["feat_ap_rssi_mean"] = np.where(
            count > 0, total / np.maximum(count, 1), np.nanmean(fallback, axis=2))
    if busy_i is not None:
        busy = corpus.temporal[:, :, :, busy_i]
        window = valid & (corpus.temporal[:, :, :, own_i] > 0.5) if own_i is not None else valid
        window = np.broadcast_to(window, busy.shape)
        count = window.sum(axis=2)
        out["feat_chan_cca_busy_frac"] = np.where(
            count > 0, np.where(window, busy, 0.0).sum(axis=2) / np.maximum(count, 1), 0.0)
    return out


def _flat_frame(corpus: TemporalCorpus, indices: np.ndarray) -> pd.DataFrame:
    observables = corpus_observables(corpus)
    rows = []
    for group_index in indices:
        for option in np.flatnonzero(corpus.option_mask[group_index]):
            row = {
                "group_id": str(corpus.group_ids[group_index]),
                "topology_id": str(corpus.topology_ids[group_index]),
                "ap_index": int(corpus.option_indices[group_index, option]),
                "label_throughput_mbps": float(corpus.labels[group_index, option]),
                "gt_n_aps": int(corpus.configured_n_aps[group_index]),
                "gt_n_hotspots": int(corpus.n_hotspots[group_index]),
                "gt_candidate_stratum": str(corpus.candidate_strata[group_index]),
                "discovery": ("full" if corpus.option_mask[group_index].sum() ==
                              corpus.configured_n_aps[group_index] else "partial"),
            }
            row.update(zip(corpus.static_features,
                           corpus.static[group_index, option].astype(float)))
            # overwrite the two columns the heuristics key on, so a baseline is
            # computed from the same observation the model was given
            for name, values in observables.items():
                row[name] = float(values[group_index, option])
            rows.append(row)
    return pd.DataFrame(rows)


def _flatten_scores(corpus: TemporalCorpus, indices: np.ndarray,
                    scores: np.ndarray) -> np.ndarray:
    return np.concatenate([
        scores[row, corpus.option_mask[group_index]]
        for row, group_index in enumerate(indices)
    ])


def _predict(model, temporal, option_mask, time_mask, static, indices, batch_size=32):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            predictions.append(model(
                torch.from_numpy(temporal[idx]),
                torch.from_numpy(option_mask[idx]),
                torch.from_numpy(time_mask[idx]),
                torch.from_numpy(static[idx]),
            ).cpu().numpy())
    return np.concatenate(predictions)


def fit_temporal_ridge(corpus: TemporalCorpus, train_idx: np.ndarray,
                       val_idx: np.ndarray, test_idx: np.ndarray
                       ) -> tuple[np.ndarray, dict]:
    """Linear full-sequence baseline: order is retained by flattened bin position."""
    from sklearn.linear_model import Ridge

    token_mask = corpus.option_mask[train_idx, :, None] & corpus.time_mask[train_idx, None, :]
    scaler = MaskedStandardizer(corpus.temporal[train_idx], token_mask)
    temporal = scaler(corpus.temporal)
    temporal *= corpus.time_mask[:, None, :, None]
    temporal *= corpus.option_mask[:, :, None, None]
    flat = temporal.reshape(temporal.shape[0], temporal.shape[1], -1)

    def rows(indices):
        return np.concatenate([flat[index, corpus.option_mask[index]] for index in indices])

    def targets(indices):
        return np.concatenate([
            corpus.labels[index, corpus.option_mask[index]] for index in indices
        ])

    def weights(indices):
        raw = np.concatenate([
            np.full(int(corpus.option_mask[index].sum()),
                    1.0 / corpus.option_mask[index].sum())
            for index in indices
        ])
        return raw / raw.mean()

    X_train, y_train = rows(train_idx), targets(train_idx)
    train_weights = weights(train_idx)
    X_val, X_test = rows(val_idx), rows(test_idx)
    val_frame = _flat_frame(corpus, val_idx)

    best = ((float("inf"), float("inf")), None)
    for log_target in (False, True):
        target = np.log1p(y_train) if log_target else y_train
        for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
            model = Ridge(alpha=alpha, solver="lsqr")
            model.fit(X_train, target, sample_weight=train_weights)
            pred = model.predict(X_val)
            if log_target:
                pred = np.expm1(np.clip(pred, -5, 12))
            metrics = selection_metrics(val_frame, pred)
            regret = metrics.get("topology_mean_regret_mbps",
                                 metrics["mean_regret_mbps"])
            key = (regret, -metrics["mean_spearman"])
            if key < best[0]:
                best = (key, (alpha, log_target))

    alpha, log_target = best[1]
    model = Ridge(alpha=alpha, solver="lsqr")
    model.fit(X_train, np.log1p(y_train) if log_target else y_train,
              sample_weight=train_weights)
    prediction = model.predict(X_test)
    if log_target:
        prediction = np.expm1(np.clip(prediction, -5, 12))
    return prediction, {
        "alpha": alpha,
        "log_target": log_target,
        "val_topology_regret_mbps": best[0][0],
        "n_flat_features": X_train.shape[1],
    }


def _loss(scores, labels, mask, objective: str, rank_temperature: float):
    if objective == "regression":
        target = torch.log1p(labels)
        loss = F.smooth_l1_loss(scores, target, reduction="none")
        per_group = (loss.masked_fill(~mask, 0.0).sum(dim=1) /
                     mask.sum(dim=1).clamp(min=1))
        return per_group.mean()

    neg = torch.finfo(scores.dtype).min
    best = labels.masked_fill(~mask, neg).max(dim=1, keepdim=True).values
    target = torch.softmax(((labels - best) / rank_temperature).masked_fill(~mask, neg), dim=1)
    log_prob = torch.log_softmax(scores.masked_fill(~mask, neg), dim=1)
    return -(target * log_prob).masked_fill(~mask, 0.0).sum(dim=1).mean()


def train_one(corpus: TemporalCorpus, train_idx: np.ndarray, val_idx: np.ndarray,
              objective: str, seed: int, epochs: int, patience: int,
              rank_temperature: float, batch_size: int, use_static: bool = True,
              shuffle_time: bool = False
              ) -> tuple[TemporalSetTransformer, MaskedStandardizer,
                         MaskedStandardizer | None, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    token_mask = corpus.option_mask[train_idx, :, None] & corpus.time_mask[train_idx, None, :]
    scaler = MaskedStandardizer(corpus.temporal[train_idx], token_mask)
    temporal = scaler(corpus.temporal)
    if shuffle_time:
        # Column 0 is time_fraction, the bin's own position. Leaving it ordered
        # keeps the positional signal intact and permutes only the measurements,
        # so the ablation removes temporal ORDER rather than the notion of time.
        shuffle_rng = np.random.default_rng(1000 + seed)
        n_steps = temporal.shape[2]
        for g in range(temporal.shape[0]):
            valid = np.flatnonzero(corpus.time_mask[g])
            for o in np.flatnonzero(corpus.option_mask[g]):
                order = shuffle_rng.permutation(valid)
                temporal[g, o, valid, 1:] = temporal[g, o, order, 1:]
    if use_static:
        static_scaler = MaskedStandardizer(
            corpus.static[train_idx], corpus.option_mask[train_idx])
        static = static_scaler(corpus.static)
    else:
        # A zero-width static block keeps every call site below unchanged while
        # giving the model literally nothing hand-engineered to lean on.
        static_scaler = None
        static = np.zeros(corpus.static.shape[:2] + (0,), dtype=np.float32)

    model = TemporalSetTransformer(
        temporal.shape[-1], temporal.shape[-2], n_static_features=static.shape[-1],
        model_dim=48, heads=4,
        temporal_layers=2, set_layers=2, dropout=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    rng = np.random.default_rng(seed)
    val_frame = _flat_frame(corpus, val_idx)

    best_key = (float("inf"), float("inf"))
    best_state = None
    stale = 0
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train_idx)
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            optimizer.zero_grad()
            scores = model(
                torch.from_numpy(temporal[idx]),
                torch.from_numpy(corpus.option_mask[idx]),
                torch.from_numpy(corpus.time_mask[idx]),
                torch.from_numpy(static[idx]),
            )
            loss = _loss(scores, torch.from_numpy(corpus.labels[idx]),
                         torch.from_numpy(corpus.option_mask[idx]), objective,
                         rank_temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        val_scores = _predict(model, temporal, corpus.option_mask, corpus.time_mask,
                              static, val_idx, batch_size)
        flat = _flatten_scores(corpus, val_idx, val_scores)
        decision_scores = np.expm1(np.clip(flat, -5, 12)) \
            if objective == "regression" else flat
        metrics = selection_metrics(val_frame, decision_scores)
        val_loss = _loss(
            torch.from_numpy(val_scores), torch.from_numpy(corpus.labels[val_idx]),
            torch.from_numpy(corpus.option_mask[val_idx]), objective,
            rank_temperature).item()
        regret = metrics.get("topology_mean_regret_mbps",
                             metrics["mean_regret_mbps"])
        key = (regret, val_loss)
        if key < best_key:
            best_key = key
            best_state = {name: value.detach().clone()
                          for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    {objective:<10} epoch={epoch + 1:3d} "
                  f"train_loss={np.mean(losses):.4f} val_regret={key[0]:.3f}")
        if stale >= patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, scaler, static_scaler, {
        "epochs": epoch + 1,
        "val_topology_regret_mbps": best_key[0],
        "val_loss": best_key[1],
        "uses_static_features": use_static,
        "time_shuffled": shuffle_time,
        "temporal_mean": scaler.mean.tolist(),
        "temporal_std": scaler.std.tolist(),
        "static_mean": static_scaler.mean.tolist() if static_scaler else [],
        "static_std": static_scaler.std.tolist() if static_scaler else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("results/temporal"))
    parser.add_argument("--repeats", type=int, default=3,
                        help="topology splits to evaluate over")
    parser.add_argument("--init-seeds", type=int, default=1,
                        help="weight initialisations per split. Until 2026-08-24 the "
                             "split seed and the init seed were the same integer, so "
                             "the reported spread mixed the two and estimated neither; "
                             "with more than one init the results file carries both "
                             "columns and the variance can be decomposed.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rank-temperature", type=float, default=5.0,
                        help="Mbps scale over which ranking targets become near-ties")
    parser.add_argument("--shuffle-time", action="store_true",
                        help="permute each option's bins in time, destroying order while "
                             "preserving the multiset of observations. If accuracy holds, "
                             "the model is aggregating, not tracking dynamics.")
    parser.add_argument("--no-static", action="store_true",
                        help="withhold the hand-built feat_* vector so the model must "
                             "learn its representation from the raw binned scan alone")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    corpus = TemporalCorpus.load(args.corpus)
    print(f"groups={len(corpus.group_ids)} topologies={len(np.unique(corpus.topology_ids))} "
          f"options={int(corpus.option_mask.sum())} shape={corpus.temporal.shape}")

    result_rows = []
    prediction_rows = []
    stratified_rows = []
    training = {}
    for repeat in range(args.repeats):
        train_idx, val_idx, test_idx = corpus.split(seed=repeat)
        print(f"\n--- split {repeat}: train={len(train_idx)} val={len(val_idx)} "
              f"test={len(test_idx)} groups ---")
        test_frame = _flat_frame(corpus, test_idx)

        ridge_prediction, ridge_info = fit_temporal_ridge(
            corpus, train_idx, val_idx, test_idx)
        predictions = {"temporal_ridge": ridge_prediction}
        training[f"split_{repeat}_temporal_ridge"] = ridge_info
        for objective in ("regression", "ranking"):
          for init in range(args.init_seeds):
            # init seed is deliberately not the split seed
            init_seed = 9000 + 17 * init
            model, scaler, static_scaler, info = train_one(
                corpus, train_idx, val_idx, objective, init_seed, args.epochs,
                args.patience, args.rank_temperature, args.batch_size,
                use_static=not args.no_static, shuffle_time=args.shuffle_time)
            temporal = scaler(corpus.temporal)
            static = (static_scaler(corpus.static) if static_scaler is not None
                      else np.zeros(corpus.static.shape[:2] + (0,), dtype=np.float32))
            scores = _predict(model, temporal, corpus.option_mask, corpus.time_mask,
                              static, test_idx, args.batch_size)
            flat = _flatten_scores(corpus, test_idx, scores)
            name = (f"temporal_transformer_{objective}" if args.init_seeds == 1
                    else f"temporal_transformer_{objective}_init{init}")
            predictions[name] = np.expm1(np.clip(flat, -5, 12)) \
                if objective == "regression" else flat
            info["init_seed"] = init_seed
            training[f"split_{repeat}_{objective}_init{init}"] = info
            torch.save({
                "state_dict": model.state_dict(),
                "objective": objective,
                "model": {
                    "n_temporal_features": corpus.temporal.shape[-1],
                    "max_steps": corpus.temporal.shape[-2],
                    "n_static_features": 0 if args.no_static else corpus.static.shape[-1],
                    "model_dim": 48,
                    "heads": 4,
                    "temporal_layers": 2,
                    "set_layers": 2,
                    "dropout": 0.1,
                },
                "temporal_features": corpus.temporal_features,
                "static_features": corpus.static_features,
                # Tensors keep the checkpoint compatible with torch.load's
                # safe weights_only=True default. NumPy arrays would require
                # opting back into unrestricted pickle loading.
                "scaler_mean": torch.from_numpy(scaler.mean.copy()),
                "scaler_std": torch.from_numpy(scaler.std.copy()),
                "static_scaler_mean": torch.from_numpy(
                    static_scaler.mean.copy() if static_scaler else np.zeros(0, np.float32)),
                "static_scaler_std": torch.from_numpy(
                    static_scaler.std.copy() if static_scaler else np.zeros(0, np.float32)),
            }, args.out_dir / f"{name}_split{repeat}.pt")


        all_predictions = {**baseline_predictions(test_frame), **predictions}
        for name, pred in all_predictions.items():
            row = {"split_seed": repeat, "model": name}
            row.update(random_selection_metrics(test_frame) if name == "random"
                       else selection_metrics(test_frame, pred))
            if name.startswith(("temporal_ridge", "temporal_transformer_regression")):
                row.update(regression_metrics(
                    test_frame["label_throughput_mbps"].to_numpy(), pred))
            result_rows.append(row)
            # Print each split as it lands. Reporting only the final aggregate
            # meant a five-split run gave no signal at all until every split had
            # finished, which is a long time to wait to discover a run is wrong.
            if name.startswith(("temporal_ridge", "temporal_transformer")):
                print(f"    {name:<38} top1={row['top1_accuracy']:.3f} "
                      f"regret={row['mean_regret_mbps']:.3f}", flush=True)

            scored = test_frame.copy()
            scored["prediction"] = pred
            for dimension in ("gt_n_aps", "gt_n_hotspots",
                              "gt_candidate_stratum", "discovery"):
                for value, subset in scored.groupby(dimension):
                    metrics = (random_selection_metrics(subset) if name == "random" else
                               selection_metrics(subset, subset["prediction"].to_numpy()))
                    stratified_rows.append({
                        "split_seed": repeat,
                        "model": name,
                        "dimension": dimension,
                        "value": value,
                        **metrics,
                    })

        keep = ["topology_id", "group_id", "ap_index", "label_throughput_mbps",
                "gt_n_aps", "gt_n_hotspots", "gt_candidate_stratum", "discovery"]
        pred_frame = test_frame[keep].copy()
        pred_frame["split_seed"] = repeat
        for name, pred in all_predictions.items():
            pred_frame[f"pred_{name}"] = pred
        prediction_rows.append(pred_frame)

    results = pd.DataFrame(result_rows)
    results.to_csv(args.out_dir / "results_raw.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        args.out_dir / "test_predictions.csv", index=False)
    pd.DataFrame(stratified_rows).to_csv(args.out_dir / "stratified.csv", index=False)
    (args.out_dir / "training.json").write_text(json.dumps(training, indent=2))

    summary = results.groupby("model")[["top1_accuracy", "mean_regret_mbps",
                                        "topology_mean_regret_mbps",
                                        "mean_regret_frac", "mean_spearman"]].agg(["mean", "std"])
    summary.to_csv(args.out_dir / "results.csv")
    print("\n=== held-out topology results ===")
    print(summary.to_string())
    print(f"\nwrote results to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
