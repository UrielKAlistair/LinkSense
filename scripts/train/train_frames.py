#!/usr/bin/env python3
"""Train the raw-frame transformer and compare it against the binned pipeline.

Experiment A of the 2026-08-24 pair. The model receives the frame trace with no
binning and no hand-built summary statistics: whether a learned encoder can
recover what the feat_* columns supply is the question, and giving it those
columns as well would answer a different one. --with-static exists only to
measure that difference deliberately.

Run:
  python scripts/train_frames.py data/combined_frames.npz \
      --out-dir results/frames_rotating --repeats 5
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
from models.frames import FrameCorpus, FrameSetTransformer, FrameStandardizer  # noqa: E402


INIT_SEED = 9013


def observables(corpus: FrameCorpus) -> dict[str, np.ndarray]:
    """Signal and occupancy per option, read off the frame trace itself.

    The heuristics have to be computed from the same observation the model got,
    or the comparison is between two different clients rather than two decision
    rules.
    """
    names = corpus.frame_features
    rssi_i, dur_i, time_i = (names.index("rssi_dbm"), names.index("duration_log1p"),
                             names.index("time_fraction"))
    n_groups, n_options = corpus.option_mask.shape
    rssi = np.full((n_groups, n_options), -100.0, dtype=np.float64)
    busy = np.zeros((n_groups, n_options), dtype=np.float64)

    from models.frames import N_RELATIONS  # noqa: F401
    for g in range(n_groups):
        valid = corpus.frame_mask[g]
        if not valid.any():
            continue
        raw = corpus.frames[g][valid]
        rel = corpus.relations[g][:, valid]
        duration_us = np.expm1(raw[:, dur_i])
        span = max(float(raw[:, time_i].max() - raw[:, time_i].min()), 1e-6)
        for o in range(n_options):
            if not corpus.option_mask[g, o]:
                continue
            from_bss = (rel[o] & 2) > 0
            if from_bss.any():
                rssi[g, o] = raw[from_bss, rssi_i].mean()
            on_channel = (rel[o] & 1) > 0
            if on_channel.any():
                busy[g, o] = duration_us[on_channel].sum() / (span * 1e6)
    return {"feat_ap_rssi_mean": rssi, "feat_chan_cca_busy_frac": busy}


def flat_frame(corpus: FrameCorpus, indices: np.ndarray,
               derived: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for g in indices:
        for o in np.flatnonzero(corpus.option_mask[g]):
            row = {
                "group_id": str(corpus.group_ids[g]),
                "topology_id": str(corpus.topology_ids[g]),
                "ap_index": int(corpus.option_indices[g, o]),
                "label_throughput_mbps": float(corpus.labels[g, o]),
                "gt_n_aps": int(corpus.configured_n_aps[g]),
                "gt_n_hotspots": int(corpus.n_hotspots[g]),
                "gt_candidate_stratum": str(corpus.candidate_strata[g]),
                "discovery": ("full" if corpus.option_mask[g].sum() ==
                              corpus.configured_n_aps[g] else "partial"),
            }
            for name, values in derived.items():
                row[name] = float(values[g, o])
            rows.append(row)
    return pd.DataFrame(rows)


def flatten_scores(corpus: FrameCorpus, indices: np.ndarray,
                   scores: np.ndarray) -> np.ndarray:
    return np.concatenate([scores[r, corpus.option_mask[g]]
                           for r, g in enumerate(indices)])


def loss_fn(scores, labels, mask, objective: str, temperature: float):
    if objective == "regression":
        loss = F.smooth_l1_loss(scores, torch.log1p(labels), reduction="none")
        return (loss.masked_fill(~mask, 0.0).sum(dim=1) /
                mask.sum(dim=1).clamp(min=1)).mean()
    neg = torch.finfo(scores.dtype).min
    best = labels.masked_fill(~mask, neg).max(dim=1, keepdim=True).values
    target = torch.softmax(((labels - best) / temperature).masked_fill(~mask, neg), dim=1)
    log_prob = torch.log_softmax(scores.masked_fill(~mask, neg), dim=1)
    return -(target * log_prob).masked_fill(~mask, 0.0).sum(dim=1).mean()


def predict(model, frames, corpus, static, indices, batch_size):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            out.append(model(
                torch.from_numpy(frames[idx]),
                torch.from_numpy(corpus.relations[idx].astype(np.int64)),
                torch.from_numpy(corpus.frame_mask[idx]),
                torch.from_numpy(corpus.option_mask[idx]),
                torch.from_numpy(static[idx])).numpy())
    return np.concatenate(out)


def train_one(corpus, frames, static, train_idx, val_idx, objective, seed,
              epochs, patience, temperature, batch_size, val_frame):
    torch.manual_seed(seed)
    model = FrameSetTransformer(frames.shape[-1], n_static_features=static.shape[-1],
                                model_dim=48, heads=4, latents=4,
                                cross_layers=2, set_layers=2, dropout=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    rng = np.random.default_rng(seed)

    best_key, best_state, stale = (float("inf"), float("inf")), None, 0
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train_idx)
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            optimizer.zero_grad()
            scores = model(
                torch.from_numpy(frames[idx]),
                torch.from_numpy(corpus.relations[idx].astype(np.int64)),
                torch.from_numpy(corpus.frame_mask[idx]),
                torch.from_numpy(corpus.option_mask[idx]),
                torch.from_numpy(static[idx]))
            loss = loss_fn(scores, torch.from_numpy(corpus.labels[idx]),
                           torch.from_numpy(corpus.option_mask[idx]),
                           objective, temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        val_scores = predict(model, frames, corpus, static, val_idx, batch_size)
        flat = flatten_scores(corpus, val_idx, val_scores)
        decision = np.expm1(np.clip(flat, -5, 12)) if objective == "regression" else flat
        metrics = selection_metrics(val_frame, decision)
        val_loss = loss_fn(torch.from_numpy(val_scores),
                           torch.from_numpy(corpus.labels[val_idx]),
                           torch.from_numpy(corpus.option_mask[val_idx]),
                           objective, temperature).item()
        key = (metrics.get("topology_mean_regret_mbps", metrics["mean_regret_mbps"]),
               val_loss)
        if key < best_key:
            best_key, stale = key, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    {objective:<10} epoch={epoch + 1:3d} "
                  f"train={np.mean(losses):.4f} val_regret={key[0]:.3f}", flush=True)
        if stale >= patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {"epochs": epoch + 1, "val_topology_regret_mbps": best_key[0],
                   "val_loss": best_key[1], "init_seed": seed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("results/frames"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rank-temperature", type=float, default=5.0)
    parser.add_argument("--with-static", action="store_true",
                        help="also give the model the hand-built feat_* vector; off by "
                             "default because the point is whether the trace alone suffices")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    corpus = FrameCorpus.load(args.corpus)
    print(f"groups={len(corpus.group_ids)} frames={corpus.frames.shape[1]} "
          f"features={corpus.frames.shape[-1]} options={int(corpus.option_mask.sum())}")
    derived = observables(corpus)

    result_rows, training = [], {}
    for repeat in range(args.repeats):
        train_idx, val_idx, test_idx = corpus.split(seed=repeat)
        print(f"\n--- split {repeat}: train={len(train_idx)} val={len(val_idx)} "
              f"test={len(test_idx)} ---", flush=True)
        val_frame = flat_frame(corpus, val_idx, derived)
        test_frame = flat_frame(corpus, test_idx, derived)

        scaler = FrameStandardizer(corpus.frames[train_idx], corpus.frame_mask[train_idx])
        frames = scaler(corpus.frames) * corpus.frame_mask[:, :, None]
        if args.with_static:
            valid = corpus.static[train_idx][corpus.option_mask[train_idx]]
            mean, std = valid.mean(0), valid.std(0)
            std[std < 1e-6] = 1.0
            static = ((corpus.static - mean) / std).astype(np.float32)
        else:
            static = np.zeros(corpus.static.shape[:2] + (0,), dtype=np.float32)

        predictions = {}
        for objective in ("regression", "ranking"):
            model, info = train_one(
                corpus, frames, static, train_idx, val_idx, objective,
                # fixed, and deliberately not derived from `repeat`: the split
                # seed and the init seed must not be the same number
                INIT_SEED, args.epochs, args.patience,
                args.rank_temperature, args.batch_size, val_frame)
            scores = predict(model, frames, corpus, static, test_idx, args.batch_size)
            flat = flatten_scores(corpus, test_idx, scores)
            name = f"frame_transformer_{objective}"
            predictions[name] = (np.expm1(np.clip(flat, -5, 12))
                                 if objective == "regression" else flat)
            training[f"split_{repeat}_{objective}"] = info
            torch.save({"state_dict": model.state_dict(), "objective": objective,
                        "frame_features": corpus.frame_features,
                        "scaler_mean": torch.from_numpy(scaler.mean.copy()),
                        "scaler_std": torch.from_numpy(scaler.std.copy())},
                       args.out_dir / f"{name}_split{repeat}.pt")

        y_test = test_frame["label_throughput_mbps"].to_numpy()
        for name, pred in {**baseline_predictions(test_frame), **predictions}.items():
            row = {"split_seed": repeat, "model": name}
            row.update(random_selection_metrics(test_frame) if name == "random"
                       else selection_metrics(test_frame, pred))
            if name == "frame_transformer_regression":
                row.update(regression_metrics(y_test, pred))
            result_rows.append(row)
            if name in predictions:
                print(f"    {name:<34} top1={row['top1_accuracy']:.3f} "
                      f"regret={row['mean_regret_mbps']:.3f}", flush=True)

    results = pd.DataFrame(result_rows)
    results.to_csv(args.out_dir / "results_raw.csv", index=False)
    (args.out_dir / "training.json").write_text(json.dumps(training, indent=2))
    summary = results.groupby("model")[["top1_accuracy", "mean_regret_mbps",
                                        "topology_mean_regret_mbps",
                                        "mean_spearman"]].agg(["mean", "std"])
    summary.to_csv(args.out_dir / "results.csv")
    print("\n=== held-out topology results ===")
    print(summary.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
