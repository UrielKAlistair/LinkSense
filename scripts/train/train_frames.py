#!/usr/bin/env python3
"""Can a model read the raw frame trace and pick a better access point?

Every other model in this project is handed a tidied-up view of what the client
heard: either summary statistics, or measurements already bucketed into fixed
time slices. This one is handed the frame list itself - one token per decoded
802.11 frame, in the order the radio saw them - and has to work out for itself
what is worth counting.

The unit of data is a scan: one client standing in one place, the trace
it recorded before it joined anything, and, for every access point it heard,
the throughput it would have got had it joined that one.

The model predicts log1p of each option's throughput, so its scores read back
as Mbps, and is reported against the heuristics a real client could run
instead.

Splits are by topology, never by scan: repeated observations of one
deployment are near-copies, so letting them straddle a split inflates every
number reported here.

Run:
  .venv/bin/python3 scripts/train/train_frames.py data/frames.npz \
      --out-dir results/frames --repeats 5
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

from scripts.models.data import MISSING_RSSI_SENTINEL  # noqa: E402
from scripts.models.evaluate import (baseline_predictions, regression_metrics,  # noqa: E402
                             selection_metrics)
from scripts.models.frames import (REL_FROM_AP, REL_ON_CHANNEL, FrameCorpus,  # noqa: E402
                           FrameSetTransformer)
from scripts.models.temporal import MaskedStandardizer  # noqa: E402


# Weight initialisation, held apart from the split seed so that the spread
# across repeats measures the split alone; reuse a split seed here and the two
# sources of variation can no longer be told apart.
INIT_SEED = 9013


def observables(corpus: FrameCorpus) -> dict[str, np.ndarray]:
    """Signal and occupancy per option, read off the frame trace itself.

    The heuristics have to be computed from the same observation the model got,
    or the comparison is between two different clients rather than two decision
    rules.
    """
    # TODO: the busy value below is decoded-frame airtime, not CCA, despite its
    # key. train_temporal.py rebuilds the same inputs in corpus_observables().
    # Decide where heuristic inputs should come from.
    names = corpus.frame_features
    rssi_i, dur_i, time_i, beacon_i = (names.index("rssi_dbm"), names.index("duration_log1p"),
                                       names.index("time_fraction"), names.index("is_beacon"))
    n_scans, n_options = corpus.option_mask.shape
    # An option with no beacon in the trace keeps the out-of-range level the
    # feature table uses for a missing reading, not a plausible one.
    rssi = np.full((n_scans, n_options), MISSING_RSSI_SENTINEL, dtype=np.float64)
    busy = np.zeros((n_scans, n_options), dtype=np.float64)

    for g in range(n_scans):
        valid = corpus.frame_mask[g]
        if not valid.any():
            continue
        raw = corpus.frames[g][valid]
        rel = corpus.relations[g][:, valid]
        duration_us = np.expm1(raw[:, dur_i])
        # time_fraction is a fraction of the window, so the window length turns
        # the span into seconds.
        span = max(float(raw[:, time_i].max() - raw[:, time_i].min())
                   * float(corpus.window_s[g]), 1e-6)
        for o in range(n_options):
            if not corpus.option_mask[g, o]:
                continue
            beacons = ((rel[o] & REL_FROM_AP) > 0) & (raw[:, beacon_i] > 0.5)
            if beacons.any():
                rssi[g, o] = raw[beacons, rssi_i].mean()
            on_channel = (rel[o] & REL_ON_CHANNEL) > 0
            if on_channel.any():
                busy[g, o] = duration_us[on_channel].sum() / (span * 1e6)
    return {"feat_ap_rssi_mean": rssi, "feat_chan_cca_busy_frac": busy}


def flat_frame(corpus: FrameCorpus, indices: np.ndarray,
               derived: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for g in indices:
        for o in np.flatnonzero(corpus.option_mask[g]):
            row = {
                "scan_id": str(corpus.scan_ids[g]),
                "topology_id": str(corpus.topology_ids[g]),
                "ap_index": int(corpus.option_indices[g, o]),
                "label_throughput_mbps": float(corpus.labels[g, o]),
                "gt_n_aps": int(corpus.configured_n_aps[g]),
                "gt_n_hotspots": int(corpus.n_hotspots[g]),
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


def loss_fn(scores, labels, mask):
    """Smooth L1 against log1p(throughput), averaged per scan then per batch."""
    loss = F.smooth_l1_loss(scores, torch.log1p(labels), reduction="none")
    return (loss.masked_fill(~mask, 0.0).sum(dim=1) /
            mask.sum(dim=1).clamp(min=1)).mean()


def predict(model, frames, corpus, indices, batch_size):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            out.append(model(
                torch.from_numpy(frames[idx]),
                torch.from_numpy(corpus.relations[idx].astype(np.int64)),
                torch.from_numpy(corpus.frame_mask[idx]),
                torch.from_numpy(corpus.option_mask[idx])).numpy())
    return np.concatenate(out)


def train_one(corpus, frames, train_idx, val_idx, seed,
              epochs, patience, batch_size, val_frame):
    torch.manual_seed(seed)
    model = FrameSetTransformer(frames.shape[-1], model_dim=48, heads=4, latents=4,
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
                torch.from_numpy(corpus.option_mask[idx]))
            loss = loss_fn(scores, torch.from_numpy(corpus.labels[idx]),
                           torch.from_numpy(corpus.option_mask[idx]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        val_scores = predict(model, frames, corpus, val_idx, batch_size)
        flat = flatten_scores(corpus, val_idx, val_scores)
        metrics = selection_metrics(val_frame, np.expm1(np.clip(flat, -5, 12)))
        val_loss = loss_fn(torch.from_numpy(val_scores),
                           torch.from_numpy(corpus.labels[val_idx]),
                           torch.from_numpy(corpus.option_mask[val_idx])).item()
        key = (metrics.get("topology_mean_regret_mbps", metrics["mean_regret_mbps"]),
               val_loss)
        if key < best_key:
            best_key, stale = key, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    epoch={epoch + 1:3d} "
                  f"train={np.mean(losses):.4f} val_regret={key[0]:.3f}", flush=True)
        if stale >= patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {"epochs": epoch + 1, "val_topology_regret_mbps": best_key[0],
                   "val_loss": best_key[1], "init_seed": seed}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("results/frames"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    corpus = FrameCorpus.load(args.corpus)
    print(f"scans={len(corpus.scan_ids)} frames={corpus.frames.shape[1]} "
          f"features={corpus.frames.shape[-1]} options={int(corpus.option_mask.sum())}")
    derived = observables(corpus)

    result_rows, training = [], {}
    for repeat in range(args.repeats):
        train_idx, val_idx, test_idx = corpus.split(seed=repeat)
        print(f"\n--- split {repeat}: train={len(train_idx)} val={len(val_idx)} "
              f"test={len(test_idx)} ---", flush=True)
        val_frame = flat_frame(corpus, val_idx, derived)
        test_frame = flat_frame(corpus, test_idx, derived)
        # k for the RSSI-minus-busy heuristic is fitted on TRAIN topologies only
        train_frame = flat_frame(corpus, train_idx, derived)

        scaler = MaskedStandardizer(corpus.frames[train_idx], corpus.frame_mask[train_idx])
        frames = scaler(corpus.frames) * corpus.frame_mask[:, :, None]

        model, info = train_one(
            corpus, frames, train_idx, val_idx, INIT_SEED,
            args.epochs, args.patience, args.batch_size, val_frame)
        scores = predict(model, frames, corpus, test_idx, args.batch_size)
        flat = flatten_scores(corpus, test_idx, scores)
        predictions = {"frame_transformer": np.expm1(np.clip(flat, -5, 12))}
        training[f"split_{repeat}"] = info
        torch.save({"state_dict": model.state_dict(),
                    "frame_features": corpus.frame_features,
                    "scaler_mean": torch.from_numpy(scaler.mean.copy()),
                    "scaler_std": torch.from_numpy(scaler.std.copy())},
                   args.out_dir / f"frame_transformer_split{repeat}.pt")

        y_test = test_frame["label_throughput_mbps"].to_numpy()
        for name, pred in {**baseline_predictions(test_frame, fit_frame=train_frame),
                           **predictions}.items():
            row = {"split_seed": repeat, "model": name}
            row.update(selection_metrics(test_frame, pred))
            if name == "frame_transformer":
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
