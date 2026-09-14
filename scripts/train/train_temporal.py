#!/usr/bin/env python3
"""Can a model that reads the whole passive scan pick a better access point?

The input is a corpus of scans. One scan is a single client
standing in one place, the sequence of time bins its radio recorded before it
joined anything, and - for every access point it heard - the throughput it
would have got had it joined that one. The job is to score the options so the
best one comes top.

A transformer is trained on that corpus with an architecture that first reads
each option's time sequence and then compares the options to each other. It
predicts log1p of each option's throughput, so its scores read back as Mbps
estimates.

It is reported against a linear model over the same flattened sequence and
against the heuristics a real client could run instead - strongest signal,
least busy channel, and signal traded off against channel occupancy. A model
that cannot beat those has not earned its complexity.

Splits are by topology, never by scan: repeated observations of one
deployment are near-copies, so letting them straddle a split inflates every
number reported here.

Run:
  .venv/bin/python3 scripts/train/train_temporal.py data/binned.npz \
      --out-dir results/tx_10ms
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

from scripts.models.evaluate import (baseline_predictions, regression_metrics,  # noqa: E402
                             selection_metrics,
                             validation_selection_key)
from scripts.models.temporal import (MaskedStandardizer, TemporalCorpus,  # noqa: E402
                             TemporalSetTransformer)


def corpus_observables(corpus: TemporalCorpus) -> dict[str, np.ndarray]:
    """Per-option signal level and channel occupancy, computed from this corpus.

    Returns the two columns the heuristic baselines key on, derived from the
    same binned observation the model is given, so that model and heuristic are
    scored on one client's measurements rather than two.

    Signal averages only the bins that carried a reading, since a bin with no
    reading has no level to average. Occupancy averages every valid bin,
    because an idle bin is a genuine zero.
    """
    # TODO: train_frames.py rebuilds the same heuristic inputs in observables().
    # Decide where heuristic inputs should come from.
    names = list(corpus.temporal_features)

    def column(*candidates):
        for name in candidates:
            if name in names:
                return names.index(name)
        return None

    rssi_i = column("option_rssi_mean")
    seen_i = column("option_observed")
    beacons_i = column("option_beacons_log1p")
    busy_i = column("cca_busy_fraction")
    # cca_busy_fraction describes whichever channel the radio was tuned to at a
    # given step, so it carries the same value for every option at that step.
    # is_option_channel marks the steps that were on this option's own channel,
    # and the average below is restricted to those. Drop that restriction and
    # every option in a scan scores identically, the baseline ties on all of
    # them, and least_busy_channel becomes a coin flip.
    own_i = column("is_option_channel")
    valid = corpus.time_mask[:, None, :]

    out = {}
    if rssi_i is not None:
        rssi = corpus.temporal[:, :, :, rssi_i]
        live = valid & (corpus.temporal[:, :, :, seen_i] > 0.5) if seen_i is not None else valid
        # Each bin's reading is itself a mean over the beacons that arrived in
        # it, so bins are weighted by that count.
        weight = (np.expm1(corpus.temporal[:, :, :, beacons_i])
                  if beacons_i is not None else np.ones_like(rssi))
        weight = np.where(live, weight, 0.0)
        total = (np.where(live, rssi, 0.0) * weight).sum(axis=2)
        heard = weight.sum(axis=2)
        # an option never heard live keeps its filled level rather than a zero
        fallback = np.where(valid, rssi, np.nan)
        out["feat_ap_rssi_mean"] = np.where(
            heard > 0, total / np.maximum(heard, 1e-9), np.nanmean(fallback, axis=2))
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
                "scan_id": str(corpus.scan_ids[group_index]),
                "topology_id": str(corpus.topology_ids[group_index]),
                "ap_index": int(corpus.option_indices[group_index, option]),
                "label_throughput_mbps": float(corpus.labels[group_index, option]),
                "gt_n_aps": int(corpus.configured_n_aps[group_index]),
                "gt_n_hotspots": int(corpus.n_hotspots[group_index]),
                "discovery": ("full" if corpus.option_mask[group_index].sum() ==
                              corpus.configured_n_aps[group_index] else "partial"),
            }
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


# One device for the whole script. The temporal encoder attends over every time
# bin of every option, so attention dominates the cost.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Weight initialisation. Held apart from the split seed so that the spread
# across repeats measures the split and not a second, tangled source of
# variation; reuse a split seed here and the two can no longer be told apart.
INIT_SEED = 9000

# Fixed so that re-running --shuffle-time reproduces the same permutation.
TIME_SHUFFLE_SEED = 20260908


def to_device(a):
    """numpy -> tensor on the training device."""
    return torch.from_numpy(a).to(DEVICE)


def _predict(model, temporal, option_mask, time_mask, indices, batch_size=32):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            predictions.append(model(
                to_device(temporal[idx]),
                to_device(option_mask[idx]),
                to_device(time_mask[idx]),
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
            key = validation_selection_key(val_frame, pred)
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


def _loss(scores, labels, mask):
    """Smooth L1 against log1p(throughput), averaged per scan then per batch.

    Averaging within a scan first stops an eight-option scan counting four times
    as much as a two-option one.
    """
    loss = F.smooth_l1_loss(scores, torch.log1p(labels), reduction="none")
    per_scan = loss.masked_fill(~mask, 0.0).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return per_scan.mean()


def train_one(corpus: TemporalCorpus, train_idx: np.ndarray, val_idx: np.ndarray,
              seed: int, epochs: int, patience: int, batch_size: int
              ) -> tuple[TemporalSetTransformer, MaskedStandardizer, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    token_mask = corpus.option_mask[train_idx, :, None] & corpus.time_mask[train_idx, None, :]
    scaler = MaskedStandardizer(corpus.temporal[train_idx], token_mask)
    # Reads corpus.temporal as main() left it. Any --shuffle-time permutation
    # is already baked into that array; permuting a local copy here would leave
    # the ridge comparator and the test-time rebuild reading ordered data.
    temporal = scaler(corpus.temporal)

    model = TemporalSetTransformer(
        temporal.shape[-1], temporal.shape[-2], model_dim=48, heads=4,
        temporal_layers=2, set_layers=2, dropout=0.1).to(DEVICE)
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
                to_device(temporal[idx]),
                to_device(corpus.option_mask[idx]),
                to_device(corpus.time_mask[idx]),
            )
            loss = _loss(scores, to_device(corpus.labels[idx]),
                         to_device(corpus.option_mask[idx]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        val_scores = _predict(model, temporal, corpus.option_mask, corpus.time_mask,
                              val_idx, batch_size)
        flat = _flatten_scores(corpus, val_idx, val_scores)
        metrics = selection_metrics(val_frame, np.expm1(np.clip(flat, -5, 12)))
        val_loss = _loss(
            to_device(val_scores), to_device(corpus.labels[val_idx]),
            to_device(corpus.option_mask[val_idx])).item()
        regret = metrics.get("topology_mean_regret_mbps",
                             metrics["mean_regret_mbps"])
        key = (regret, val_loss)
        if key < best_key:
            best_key = key
            # .cpu() so the checkpoint stays loadable on a machine without a GPU
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    epoch={epoch + 1:3d} "
                  f"train_loss={np.mean(losses):.4f} val_regret={key[0]:.3f}")
        if stale >= patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, scaler, {
        "epochs": epoch + 1,
        "val_topology_regret_mbps": best_key[0],
        "val_loss": best_key[1],
        "temporal_mean": scaler.mean.tolist(),
        "temporal_std": scaler.std.tolist(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("results/temporal"))
    parser.add_argument("--repeats", type=int, default=3,
                        help="topology splits to evaluate over")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shuffle-time", action="store_true",
                        help="permute each option's bins in time, destroying order while "
                             "preserving the multiset of observations. If accuracy holds, "
                             "the model is aggregating, not tracking dynamics.")
    return parser.parse_args()


def shuffle_time_bins(corpus: TemporalCorpus) -> None:
    """Permute each scan's time bins in place, destroying order and nothing else.

    One permutation per group, reused for every option in it. A fresh
    permutation per option would break the alignment between options sharing a
    bin, and would let several options each claim the radio sat on their own
    channel at the same instant.

    Column 0, time_fraction, stays in ascending order: a bin still knows where
    it sits in the window but not what preceded it. Every other column moves,
    the *_age columns included, so each bin carries its own recency as a value.

    Mutates corpus.temporal before anything reads it, so training, the ridge
    comparator and the test-time rebuild all see the same permuted array.
    """
    if corpus.temporal_features[0] != "time_fraction":
        raise ValueError(
            "--shuffle-time holds column 0 fixed and expects it to be "
            f"time_fraction; this corpus has {corpus.temporal_features[0]}")
    rng = np.random.default_rng(TIME_SHUFFLE_SEED)
    for group in range(corpus.temporal.shape[0]):
        valid = np.flatnonzero(corpus.time_mask[group])
        order = rng.permutation(valid)
        corpus.temporal[group][:, valid, 1:] = corpus.temporal[group][:, order, 1:]
    print(f"TIME SHUFFLED: one permutation per group over "
          f"{corpus.temporal.shape[2]} steps, time_fraction held in order")


def save_checkpoint(path: Path, model, scaler, corpus: TemporalCorpus) -> None:
    """Write one trained model plus everything needed to score with it again."""
    torch.save({
        "state_dict": model.state_dict(),
        "model": {
            "n_temporal_features": corpus.temporal.shape[-1],
            "max_steps": corpus.temporal.shape[-2],
            "model_dim": 48,
            "heads": 4,
            "temporal_layers": 2,
            "set_layers": 2,
            "dropout": 0.1,
        },
        "temporal_features": corpus.temporal_features,
        # Tensors keep the checkpoint loadable under torch.load's safe
        # weights_only=True default; NumPy arrays would need unrestricted pickle.
        "scaler_mean": torch.from_numpy(scaler.mean.copy()),
        "scaler_std": torch.from_numpy(scaler.std.copy()),
    }, path)


def train_split(corpus: TemporalCorpus, repeat: int, args, training: dict):
    """Fit every model on one topology split; returns its predictions and frames."""
    train_idx, val_idx, test_idx = corpus.split(seed=repeat)
    print(f"\n--- split {repeat}: train={len(train_idx)} val={len(val_idx)} "
          f"test={len(test_idx)} groups ---")
    test_frame = _flat_frame(corpus, test_idx)
    # k for the RSSI-minus-busy heuristic is fitted on TRAIN topologies only
    train_frame = _flat_frame(corpus, train_idx)

    ridge_prediction, ridge_info = fit_temporal_ridge(
        corpus, train_idx, val_idx, test_idx)
    predictions = {"temporal_ridge": ridge_prediction}
    training[f"split_{repeat}_temporal_ridge"] = ridge_info

    model, scaler, info = train_one(
        corpus, train_idx, val_idx, INIT_SEED, args.epochs, args.patience,
        args.batch_size)
    temporal = scaler(corpus.temporal)
    scores = _predict(model, temporal, corpus.option_mask, corpus.time_mask,
                      test_idx, args.batch_size)
    flat = _flatten_scores(corpus, test_idx, scores)
    predictions["temporal_transformer"] = np.expm1(np.clip(flat, -5, 12))
    info["init_seed"] = INIT_SEED
    training[f"split_{repeat}"] = info
    save_checkpoint(args.out_dir / f"temporal_transformer_split{repeat}.pt",
                    model, scaler, corpus)
    return predictions, test_frame, train_frame


def score_split(all_predictions: dict, test_frame, repeat: int, args):
    """Headline and stratified metric rows for one split."""
    provenance = {"split_seed": repeat, "time_shuffled": args.shuffle_time}
    results, stratified = [], []
    for name, pred in all_predictions.items():
        row = {**provenance, "model": name}
        row.update(selection_metrics(test_frame, pred))
        if name.startswith(("temporal_ridge", "temporal_transformer_regression")):
            row.update(regression_metrics(
                test_frame["label_throughput_mbps"].to_numpy(), pred))
        results.append(row)
        # report each split as it finishes rather than only at the end
        if name.startswith(("temporal_ridge", "temporal_transformer")):
            print(f"    {name:<38} top1={row['top1_accuracy']:.3f} "
                  f"regret={row['mean_regret_mbps']:.3f}", flush=True)

        scored = test_frame.copy()
        scored["prediction"] = pred
        for dimension in ("gt_n_aps", "gt_n_hotspots", "discovery"):
            for value, subset in scored.groupby(dimension):
                metrics = selection_metrics(subset, subset["prediction"].to_numpy())
                stratified.append({**provenance, "model": name,
                                   "dimension": dimension, "value": value, **metrics})
    return results, stratified


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    corpus = TemporalCorpus.load(args.corpus)
    print(f"device={DEVICE}"
          + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
    print(f"scans={len(corpus.scan_ids)} topologies={len(np.unique(corpus.topology_ids))} "
          f"options={int(corpus.option_mask.sum())} shape={corpus.temporal.shape}")
    if args.shuffle_time:
        shuffle_time_bins(corpus)

    result_rows, stratified_rows, prediction_rows, training = [], [], [], {}
    for repeat in range(args.repeats):
        predictions, test_frame, train_frame = train_split(
            corpus, repeat, args, training)
        all_predictions = {**baseline_predictions(test_frame, fit_frame=train_frame),
                           **predictions}
        rows, stratified = score_split(all_predictions, test_frame, repeat, args)
        result_rows += rows
        stratified_rows += stratified

        keep = ["topology_id", "scan_id", "ap_index", "label_throughput_mbps",
                "gt_n_aps", "gt_n_hotspots", "discovery"]
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
                                        "mean_regret_frac",
                                        "mean_spearman"]].agg(["mean", "std"])
    summary.to_csv(args.out_dir / "results.csv")
    print("\n=== held-out topology results ===")
    print(summary.to_string())
    print(f"\nwrote results to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
