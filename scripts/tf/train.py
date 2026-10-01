#!/usr/bin/env python3
"""Train a cell model on the cached scans, and score it on held-out topologies
against the reference rules.

INPUT
  cache_dir, written by cache_dataset.py: <scan_id>.npz per scan, and _index.npz.
  --evaluate-only, to score the models already saved in out_dir rather than
  train new ones: the folds are cut the same way, so each model is scored on
  the topologies it was held out from.
  --model, which model: joint_ap (joint_ap_model.py) scores every discovered
  AP of a scan in one pass, target_ap (target_ap_model.py) one AP per pass, binned
  (binned_model.py) reads only the cells' aggregates, as one sequence of dwells
  per AP, and llm (llm_model.py) is joint_ap with a frozen language model
  reading the grid. All four read the same batches and predict the same things.
  --only-fold, to run one rotation instead of all of them, leaving the deal
  itself untouched. Its numbers compare against the same fold of a full run,
  never against that run's average over folds.

OUTPUT, in out_dir (results/<model> unless given)
  results_raw.csv   one row per fold and per model or reference rule
  summary.csv       each metric's mean and standard deviation across folds
  training.json     per fold: the epoch kept and the validation curve
  fold<k>.pt        the model kept for fold k, with the statistics its inputs
                    were standardised by

PROCESS, per fold
  1. Take the scans of fold k as the test set and of fold k+1 as the
     validation set, the rest training: whole topologies, 60/20/20, dealt
     within each deployment's AP count. Over the run every topology is
     tested exactly once.
  2. Fit every standardisation on the training scans alone.
  3. Train for a fixed number of epochs, the learning rate falling along a
     cosine to zero: squared error for the first epochs, then Gaussian negative
     log likelihood. Keep the epoch whose predictions give the lowest regret on
     the validation scans, Spearman breaking ties.
  4. Predict each test AP's throughput with an interval, and score it and the
     reference rules with common.evaluate.

Run:
  .venv/bin/python3 scripts/tf/train.py data/cache \
      --model joint_ap --out results/folds/joint_ap

  .venv/bin/python3 scripts/tf/train.py data/cache \
      --model joint_ap --grid-reader llm --only-fold 0 --out results/llm/joint
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.tf.binned_model import BinnedModel  # noqa: E402
from scripts.tf.cache_dataset import NO_AP  # noqa: E402
from scripts.tf.joint_ap_model import JointAPModel  # noqa: E402
from scripts.tf.layers import TAG_MODES, CellBatch  # noqa: E402
from scripts.tf.target_ap_model import TargetAPModel  # noqa: E402
from scripts.common.evaluate import (LABEL_COL, SCAN_COL, all_metrics,  # noqa: E402
                                     selection_metrics, validation_selection_key)
from scripts.common.splits import N_FOLDS, split_rows_by_topology  # noqa: E402

CellModel = JointAPModel | TargetAPModel | BinnedModel

# TODO: neither of the two budgets this file spends has been measured.
#   1. Epochs. The runs so far kept epochs 14 to 47 of 60, so the cosine's
#      last stretch has never produced the epoch that was kept. Sweep --epochs
#      to find where validation error stops improving.
#   2. Training data. The corpus has never been cut down to see how much of it
#      the models need: whether 15% of the topologies already reaches this
#      accuracy, or whether the curve is still climbing at 100% and more
#      simulation would pay. That needs a --train-fraction here, as
#      baselines/train.py has, subsetting whole topologies within each AP count.

# Epochs trained on squared error before the loss adds the predicted variance.
WARMUP_EPOCHS = 5

# What the head's outputs mean and what is fitted to them.
OBJECTIVES = ("gaussian", "ranking")

# A model trained from scratch and a frozen backbone under adapters want
# different optimiser settings, so --lr and --weight-decay default per model
# rather than to one pair that would be an order of magnitude out for one.
@dataclasses.dataclass(frozen=True)
class OptimiserDefaults:
    """The learning rate and weight decay a model is trained under."""

    lr: float
    weight_decay: float


FROM_SCRATCH_OPTIMISER = OptimiserDefaults(lr=1e-3, weight_decay=1e-2)
LLM_OPTIMISER = OptimiserDefaults(lr=1e-4, weight_decay=1e-4)

# Standard normal quantiles bounding the central 50% and 90% of a prediction.
Z_50, Z_90 = 0.6745, 1.6449


def run_fold(cache: Cache, fold: int, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    """Train and score one fold: one row per model and reference rule, and the
    record of its training."""
    # 1. Split by topology, this fold testing and the next validating.
    train_scans, val_scans, test_scans = split_rows_by_topology(
        cache.topology_ids, cache.configured_n_aps, test_fold=fold, n_folds=args.folds)

    # 2 and 3. Standardise on the training scans and train, or, with
    #          --evaluate-only, take back the model this fold trained before
    #          along with the statistics it was standardised under.
    if args.evaluate_only:
        model, scale, record = load_fold(cache, fold, args)
    else:
        scale = Scale([cache.scans[i] for i in train_scans])
        # The fold index doubles as the seed for the initial weights and for
        # the frames each batch samples, so a fold is reproducible on its own.
        # Seeding precedes build_model, which draws every initial weight.
        torch.manual_seed(fold)
        model = build_model(cache, args)
        record = train(model, cache, train_scans, val_scans, scale, args,
                       np.random.default_rng(fold))
        torch.save({"model": args.model, "grid_reader": args.grid_reader,
                    "objective": args.objective,
                    "state_dict": trainable_state(model),
                    "scale": dataclasses.asdict(scale.state()),
                    "frame_names": cache.frame_names,
                    "aggregate_names": cache.aggregate_names,
                    "descriptor_names": cache.descriptor_names,
                    "backbone": args.backbone,
                    "random_backbone": args.random_backbone,
                    "lora_rank": args.lora_rank},
                   args.out / f"fold{fold}.pt")

    # 4. Predict the test APs, and score them against the reference rules.
    mu, log_var = predict(model, cache, test_scans, scale, args)
    results = score(cache, test_scans, mu, log_var, scale, args.model, args.objective)
    record.update(train=len(train_scans), val=len(val_scans), test=len(test_scans))
    return results, record


def build_model(cache: Cache, args: argparse.Namespace) -> CellModel:
    """An untrained model of the kind --model names, sized to the cache."""
    sizes = (len(cache.frame_names), len(cache.aggregate_names), len(cache.descriptor_names))
    if args.model == "binned":
        return BinnedModel(len(cache.aggregate_names)).to(args.device)

    reader = None
    if args.grid_reader == "llm":
        # imported here rather than at module scope: only this reader needs
        # transformers and peft, which cost seconds to import on every run.
        from scripts.tf.llm_model import llama_reader
        reader = llama_reader(args.backbone, args.lora_rank,
                              pretrained=not args.random_backbone,
                              gradient_checkpointing=args.llm_gradient_checkpointing)
    # The backbone is what the encoder blocks would have been, so a model reading
    # the grid with one builds none of its own.
    shared = dict(readout_bias=not args.no_readout_bias,
                  encoder_bias=not args.no_encoder_bias, tag_mode=args.tag_mode,
                  tag_scale=args.tag_scale, grid_reader=reader,
                  layers=0 if reader else 2)
    if args.model == "joint_ap":
        return JointAPModel(*sizes, n_identities=args.identities,
                            ap_identity_tag=not args.no_identity_tag,
                            **shared).to(args.device)
    return TargetAPModel(*sizes, role_tag=not args.no_identity_tag,
                         fixed_role_codes=args.fixed_role_codes,
                         **shared).to(args.device)


def frozen_parameters(model: CellModel) -> set[str]:
    """The names of the parameters that do not train.

    Empty for the models trained from scratch. For the llm model it is the whole
    backbone, which is what a checkpoint leaves out and what a load is allowed to
    find missing.
    """
    return {name for name, p in model.named_parameters() if not p.requires_grad}


def trainable_state(model: CellModel) -> dict:
    """What a checkpoint has to carry: all of the state except frozen parameters.

    For the three models trained from scratch that is the whole state dict. The
    llm model's backbone is frozen and comes back from its own checkpoint when
    the model is rebuilt, so writing it here would add about 2 GB a fold and
    restore nothing. Buffers are kept whether or not their module is frozen.
    """
    frozen = frozen_parameters(model)
    return {k: v for k, v in model.state_dict().items() if k not in frozen}


def load_fold(cache: Cache, fold: int, args: argparse.Namespace
              ) -> tuple[CellModel, Scale, dict]:
    """The model a previous run trained on this fold, and the statistics it
    standardised its inputs by.

    The scale comes back from the checkpoint rather than being refitted on the
    training scans, and the cache's columns are checked against the ones the
    model was trained on: a cache rebuilt since would otherwise be scored
    through a scale fitted to columns it no longer holds.
    """
    path = args.out / f"fold{fold}.pt"
    checkpoint = torch.load(path, map_location=args.device)
    if checkpoint["model"] != args.model:
        raise SystemExit(f"{path} holds a {checkpoint['model']} model, "
                         f"and --model asks for {args.model}")
    stored = (checkpoint["frame_names"], checkpoint["aggregate_names"],
              checkpoint["descriptor_names"])
    if stored != (cache.frame_names, cache.aggregate_names, cache.descriptor_names):
        raise SystemExit(f"{path} was trained on a cache with different columns "
                         f"than {args.cache_dir}; retrain rather than score it")
    if checkpoint.get("objective", "gaussian") != args.objective:
        raise SystemExit(
            f"{path} was trained under the "
            f"{checkpoint.get('objective', 'gaussian')} objective and this run "
            f"asks for {args.objective}; its outputs do not mean the same thing")
    if checkpoint.get("grid_reader", "native") != args.grid_reader:
        raise ValueError(f"{path} was trained with the "
                         f"{checkpoint.get('grid_reader', 'native')} grid reader "
                         f"and --grid-reader asks for {args.grid_reader}")
    if args.grid_reader == "llm":
        # The backbone is frozen, so it is absent from the checkpoint and a load
        # cannot tell which one trained the adapters. Scoring them against a
        # different backbone, or against random weights when they were trained on
        # pretrained ones, would load cleanly and report a meaningless number.
        trained_under = tuple(checkpoint.get(key) for key in
                              ("backbone", "random_backbone", "lora_rank"))
        asked_for = (args.backbone, args.random_backbone, args.lora_rank)
        if trained_under != asked_for:
            raise SystemExit(
                f"{path} trained its adapters on backbone, random and rank "
                f"{trained_under}, and this run asks for {asked_for}")

    model = build_model(cache, args)
    # Frozen parameters are left out of a checkpoint on purpose, so they are the
    # only keys allowed to be missing from it. Anything else missing, or anything
    # in it this model has no home for, means it is the wrong checkpoint.
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    unaccounted = sorted(set(missing) - frozen_parameters(model))
    if unaccounted or unexpected:
        raise SystemExit(
            f"{path} does not fit this model: {len(unaccounted)} of its "
            f"parameters are missing {unaccounted[:3]} and {len(unexpected)} of "
            f"its entries are unexpected {list(unexpected)[:3]}")
    print(f"    scoring the model saved in {path}", flush=True)
    return model, Scale.restore(checkpoint["scale"]), {"loaded": str(path)}


# ---------------------------------------------------------------------------
# The cache, in memory
# ---------------------------------------------------------------------------

# The frame table's one column that places a frame rather than describing it.
# It is split off as each scan is read, and reaches the model unstandardised.
OFFSET_FEATURE = "offset_in_dwell"

@dataclasses.dataclass
class CachedScan:
    """One scan as cache_dataset.py wrote it, with each cell's rows of the frame
    table found once."""

    scan_id: str
    topology_id: str
    frames: np.ndarray            # frame table less its offsets, one row per frame
    frame_offsets: np.ndarray     # where in its dwell each frame began, 0 to 1
    frame_categories: np.ndarray
    members: list[np.ndarray]     # each cell's rows of frames, in the order heard
    cell_is_channel: np.ndarray
    cell_dwell: np.ndarray
    cell_channel: np.ndarray
    cell_ap: np.ndarray           # AP number, or NO_AP on a channel cell
    cell_aggregates: np.ndarray
    cca_samples: np.ndarray       # one row per dwell
    dwell_channel: np.ndarray
    descriptors: np.ndarray       # one row per discovered AP, in AP-number order
    labels: np.ndarray
    ap_channels: np.ndarray


class Cache:
    """Every cached scan in memory, in the index's order."""

    def __init__(self, cache_dir: Path):
        index = np.load(cache_dir / "_index.npz")
        frame_names = index["frame_features"].tolist()
        offset_column = frame_names.index(OFFSET_FEATURE)
        self.frame_names = [name for name in frame_names if name != OFFSET_FEATURE]
        self.aggregate_names = index["cell_aggregates"].tolist()
        self.descriptor_names = index["descriptor_features"].tolist()
        self.topology_ids = index["topology_ids"]
        self.configured_n_aps = index["configured_n_aps"]
        self.scans = [read_scan(cache_dir / f"{scan_id}.npz", scan_id, topology_id,
                                offset_column)
                      for scan_id, topology_id in zip(index["scan_ids"], self.topology_ids)]


def read_scan(path: Path, scan_id: str, topology_id: str,
              offset_column: int) -> CachedScan:
    """One scan's cache file, with its cells' members read back from the frame
    tags by the rule Cell.find_frames wrote them by, and its frames' offsets
    split off the frame table."""
    with np.load(path) as file:
        cached = {key: file[key] for key in file.files}
    frames = cached["frames"]
    frame_dwell, frame_ap = cached["frame_dwell"], cached["frame_ap"]
    members = []
    for dwell, ap in zip(cached["cell_dwell"], cached["cell_ap"]):
        mine = frame_dwell == dwell
        if ap != NO_AP:
            mine &= frame_ap == ap
        members.append(np.flatnonzero(mine))
    return CachedScan(
        scan_id=str(scan_id), topology_id=str(topology_id),
        frames=np.delete(frames, offset_column, axis=1),
        frame_offsets=frames[:, offset_column], frame_categories=cached["frame_categories"],
        members=members, cell_is_channel=cached["cell_is_channel"],
        cell_dwell=cached["cell_dwell"],
        cell_channel=cached["cell_channel"], cell_ap=cached["cell_ap"].astype(np.int64),
        cell_aggregates=cached["cell_aggregates"], cca_samples=cached["cca_samples"],
        dwell_channel=cached["dwell_channel"], descriptors=cached["descriptors"],
        labels=cached["labels"], ap_channels=cached["ap_channels"])


# ---------------------------------------------------------------------------
# 2. Standardise on the training scans
# ---------------------------------------------------------------------------

class ColumnScale:
    """Each column's mean and standard deviation over the rows of many arrays.

    A column with no spread keeps a scale of 1: it is centred, not blown up.
    """

    def __init__(self, arrays: list[np.ndarray]):
        rows, total, squares = 0, 0.0, 0.0
        for array in arrays:
            values = array.astype(np.float64)
            rows += len(values)
            total = total + values.sum(axis=0)
            squares = squares + (values ** 2).sum(axis=0)
        self.mean = total / rows
        self.std = np.sqrt(np.maximum(squares / rows - self.mean ** 2, 0.0))
        self.std[self.std < 1e-6] = 1.0

    @classmethod
    def of(cls, mean: list, std: list) -> "ColumnScale":
        """A ColumnScale with its statistics already measured, as a checkpoint
        stores them, rather than measured here from arrays."""
        scale = cls.__new__(cls)
        scale.mean, scale.std = np.asarray(mean), np.asarray(std)
        return scale

    def __call__(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)


@dataclasses.dataclass
class ScaleState:
    """What a Scale holds, as plain lists, for the checkpoint."""

    frames: tuple[list, list]
    channel_cells: tuple[list, list]
    ap_cells: tuple[list, list]
    descriptors: tuple[list, list]
    target: tuple[float, float]


class Scale:
    """The training scans' statistics, applied to any scan.

    Channel cells and AP cells are scaled apart: they share their columns but
    not what fills them, since each leaves empty the columns that belong to the
    other. The target is log1p of throughput, standardised.
    """

    def __init__(self, scans: list[CachedScan]):
        self.frames = ColumnScale([scan.frames for scan in scans])
        self.channel_cells = ColumnScale(
            [scan.cell_aggregates[scan.cell_is_channel] for scan in scans])
        self.ap_cells = ColumnScale(
            [scan.cell_aggregates[~scan.cell_is_channel] for scan in scans])
        self.descriptors = ColumnScale([scan.descriptors for scan in scans])
        targets = np.log1p(np.concatenate([scan.labels for scan in scans]).astype(np.float64))
        self.target_mean, self.target_std = float(targets.mean()), float(targets.std())

    def aggregates(self, scan: CachedScan) -> np.ndarray:
        channel = scan.cell_is_channel
        out = np.empty(scan.cell_aggregates.shape, dtype=np.float32)
        out[channel] = self.channel_cells(scan.cell_aggregates[channel])
        out[~channel] = self.ap_cells(scan.cell_aggregates[~channel])
        return out

    def target(self, labels: np.ndarray) -> np.ndarray:
        return ((np.log1p(labels) - self.target_mean) / self.target_std).astype(np.float32)

    def log1p_throughput(self, standardised: np.ndarray) -> np.ndarray:
        return standardised * self.target_std + self.target_mean

    def state(self) -> ScaleState:
        pair = lambda s: (s.mean.tolist(), s.std.tolist())  # noqa: E731
        return ScaleState(pair(self.frames), pair(self.channel_cells), pair(self.ap_cells),
                          pair(self.descriptors), (self.target_mean, self.target_std))

    @classmethod
    def restore(cls, state: dict) -> "Scale":
        """A Scale from a checkpoint's stored statistics, for scoring a model
        without the scans it was fitted on."""
        scale = cls.__new__(cls)
        for name in ("frames", "channel_cells", "ap_cells", "descriptors"):
            setattr(scale, name, ColumnScale.of(*state[name]))
        scale.target_mean, scale.target_std = state["target"]
        return scale


# ---------------------------------------------------------------------------
# 3. Train, a batch of scans at a time
# ---------------------------------------------------------------------------

def train(model: CellModel, cache: Cache, train_scans: np.ndarray,
          val_scans: np.ndarray, scale: Scale, args: argparse.Namespace,
          rng: np.random.Generator) -> dict:
    """Fit the model for args.epochs epochs, and leave it holding the epoch
    whose means give the lowest validation regret, Spearman breaking ties.

    The learning rate falls from args.lr to zero along a cosine, one step per
    batch. Only epochs that trained on the full likelihood are kept, so the
    model kept has a variance that was trained.
    """
    # Only what trains: the llm model's backbone is frozen, and handing its
    # parameters to AdamW would have it carry moment estimates for a billion of
    # them.
    training = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(training, lr=args.lr,
                                  weight_decay=args.weight_decay)
    batches = math.ceil(len(train_scans) / args.batch_size)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * batches)
    best_key, best_epoch, best_state, curve = (math.inf, math.inf), None, None, []
    for epoch in range(args.epochs):
        started = time.time()
        model.train()
        # The warm-up exists so the variance cannot widen before the mean is
        # worth anything. A ranking run predicts no variance, so it has none.
        ranking = args.objective == "ranking"
        with_variance = epoch >= WARMUP_EPOCHS
        keepable = ranking or with_variance
        losses = []
        order = rng.permutation(train_scans)
        for start in range(0, len(order), args.batch_size):
            batch, target, mask = make_batch(
                [cache.scans[i] for i in order[start:start + args.batch_size]],
                scale, args, rng)
            with autocast(args.device):
                mu, log_var = model(batch)
            loss = (listnet_loss(mu.float(), target, mask, args.rank_temperature)
                    if ranking else
                    gaussian_nll(mu.float(), log_var.float(), target, mask, with_variance))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(training, 1.0)
            optimizer.step()
            schedule.step()
            losses.append(loss.item())

        mu, _ = predict(model, cache, val_scans, scale, args)
        # The epoch kept is the one that decides best, ordered as the baselines
        # order their grid. Only the ranking within a scan is read, so the
        # standardised means serve as scores without being turned back to Mbps.
        key = validation_selection_key(ap_frame(cache, val_scans), np.concatenate(mu))
        curve.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                      "val_regret_mbps": key[0], "val_spearman": -key[1],
                      "lr": schedule.get_last_lr()[0],
                      "seconds": round(time.time() - started, 1)})
        print(f"    epoch {epoch:3d}  train {np.mean(losses):8.4f}  "
              f"val regret {key[0]:.4f}  {time.time() - started:5.1f}s", flush=True)
        if keepable and key < best_key:
            best_key, best_epoch = key, epoch
            # What trains, and not the frozen backbone beside it: keeping the
            # whole state dict here puts a second copy of the backbone on the
            # device every time validation improves.
            best_state = {k: v.detach().clone()
                          for k, v in trainable_state(model).items()}

    # Frozen parameters are absent from best_state and did not move, so they are
    # the only keys a load is allowed to find missing.
    model.load_state_dict(best_state, strict=False)
    return {"best_epoch": best_epoch, "best_val_regret_mbps": best_key[0],
            "best_val_spearman": -best_key[1], "curve": curve}


def autocast(device: str):
    """bfloat16 autocast on a GPU; nothing on the CPU."""
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda"))


def gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor,
                 mask: torch.Tensor, variance: bool = True) -> torch.Tensor:
    """Gaussian negative log likelihood of the predicted means and log
    variances, averaged over every discovered AP of the batch.

    Each AP is one prediction and carries the same weight, so a scan pulls on
    the fit in proportion to how many APs it holds. The mask leaves the padded
    APs out of both the sum and the count. With `variance` false this is
    plain squared error, which is how training starts: early on a model can cut
    its loss by widening the variance rather than by predicting better, and
    never recovers the mean it gave up.
    """
    if variance:
        loss = 0.5 * (log_var + (target - mu) ** 2 * torch.exp(-log_var))
    else:
        loss = 0.5 * (target - mu) ** 2
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def listnet_loss(score: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                 temperature: float) -> torch.Tensor:
    """Cross-entropy between the scan's predicted choice distribution and the
    one its labels imply, averaged over scans.

    The head's first output is read as a score rather than a throughput: a
    softmax over a scan's discovered APs turns the scores into the probability
    of each being the best, and the labels are put through the same softmax to
    say what that probability should have been. Padded APs are barred from both
    distributions. `temperature` sets how sharp the target is; at 0 it is
    one-hot on the best AP, and a tie shares the mass evenly.

    Nothing here reads the scale of a score, only its order within one scan, so
    the number a model returns is no longer a throughput and the regression
    metrics do not apply to it.
    """
    barred = torch.finfo(score.dtype).min
    predicted = torch.log_softmax(score.masked_fill(~mask, barred), dim=-1)
    if temperature > 0:
        wanted = torch.softmax((target / temperature).masked_fill(~mask, barred), dim=-1)
    else:
        best = target.masked_fill(~mask, -torch.inf).max(dim=-1, keepdim=True).values
        ties = (target >= best) & mask
        wanted = ties.to(score.dtype) / ties.sum(dim=-1, keepdim=True).clamp(min=1)
    # A scan with nothing to choose between teaches nothing, and its row of the
    # softmax is degenerate rather than wrong.
    scored = mask.sum(dim=-1) > 1
    per_scan = -(wanted * predicted).sum(dim=-1)
    return (per_scan * scored).sum() / scored.sum().clamp(min=1)


def make_batch(scans: list[CachedScan], scale: Scale, args: argparse.Namespace,
               rng: np.random.Generator | None) -> tuple[CellBatch, torch.Tensor, torch.Tensor]:
    """Scans padded into one CellBatch, with their standardised targets and
    which APs are real.

    Given rng, the batch is for training: a cell holding more than
    args.frame_cap frames keeps a random subset of them, and the APs take
    identities drawn at random. Without it, the cell keeps an evenly spaced
    subset, and the APs take identities in AP-number order.
    """
    n_scans = len(scans)
    n_cells = max(len(scan.members) for scan in scans)
    n_aps = max(len(scan.labels) for scan in scans)
    n_members = max(1, min(args.frame_cap,
                           max(len(rows) for scan in scans for rows in scan.members)))

    # One frame table for the batch, the scans laid end to end, and the row
    # each scan's frames start at.
    frames = np.concatenate([scale.frames(scan.frames) for scan in scans])
    offsets = np.concatenate([scan.frame_offsets for scan in scans]).astype(np.float32)
    categories = np.concatenate([scan.frame_categories for scan in scans]).astype(np.int64)
    starts = np.cumsum([0] + [len(scan.frames) for scan in scans[:-1]])
    members = np.zeros((n_scans, n_cells, n_members), np.int64)
    member_mask = np.zeros((n_scans, n_cells, n_members), bool)
    aggregates = np.zeros((n_scans, n_cells, scans[0].cell_aggregates.shape[1]), np.float32)
    cca = np.zeros((n_scans, n_cells, scans[0].cca_samples.shape[1]), np.float32)
    cell_is_channel = np.zeros((n_scans, n_cells), bool)
    dwell, channel, cell_identity = (np.zeros((n_scans, n_cells), np.int64)
                                     for _ in range(3))
    cell_mask = np.zeros((n_scans, n_cells), bool)
    descriptors = np.zeros((n_scans, n_aps, scans[0].descriptors.shape[1]), np.float32)
    ap_channel, ap_identity = (np.zeros((n_scans, n_aps), np.int64) for _ in range(2))
    ap_mask = np.zeros((n_scans, n_aps), bool)
    target = np.zeros((n_scans, n_aps), np.float32)

    for g, scan in enumerate(scans):
        for c, rows in enumerate(scan.members):
            kept = subsample(rows, n_members, rng)
            members[g, c, :len(kept)] = starts[g] + kept
            member_mask[g, c, :len(kept)] = True

        cells = len(scan.members)
        aggregates[g, :cells] = scale.aggregates(scan)
        channel_cells = scan.cell_is_channel
        cca[g, :cells][channel_cells] = scan.cca_samples[scan.cell_dwell[channel_cells]]
        cell_is_channel[g, :cells] = scan.cell_is_channel
        dwell[g, :cells] = scan.cell_dwell
        channel[g, :cells] = scan.cell_channel
        cell_mask[g, :cells] = True

        aps = len(scan.labels)
        identities = (rng.permutation(args.identities)[:aps] if rng is not None
                      else np.arange(aps))
        cell_identity[g, :cells] = np.where(scan.cell_ap != NO_AP,
                                            identities[scan.cell_ap], 0)
        descriptors[g, :aps] = scale.descriptors(scan.descriptors)
        ap_channel[g, :aps], ap_identity[g, :aps] = scan.ap_channels, identities
        ap_mask[g, :aps] = True
        target[g, :aps] = scale.target(scan.labels)

    to = lambda array: torch.from_numpy(array).to(args.device)  # noqa: E731
    batch = CellBatch(
        frames=to(frames), frame_offsets=to(offsets), frame_categories=to(categories),
        cell_members=to(members), cell_member_mask=to(member_mask),
        cell_aggregates=to(aggregates), cell_cca=to(cca),
        cell_is_channel=to(cell_is_channel), cell_dwell=to(dwell),
        cell_channel=to(channel), cell_ap_identity=to(cell_identity),
        cell_mask=to(cell_mask), descriptors=to(descriptors),
        ap_channel=to(ap_channel), ap_identity=to(ap_identity), ap_mask=to(ap_mask))
    return batch, to(target), to(ap_mask)


def subsample(rows: np.ndarray, limit: int, rng: np.random.Generator | None) -> np.ndarray:
    """At most `limit` of a cell's rows, in the order heard: a random subset
    given rng, an evenly spaced one without."""
    if len(rows) <= limit:
        return rows
    if rng is not None:
        return np.sort(rng.choice(rows, limit, replace=False))
    return rows[np.linspace(0, len(rows) - 1, limit).round().astype(np.int64)]


# ---------------------------------------------------------------------------
# 4. Predict and score
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model: CellModel, cache: Cache, positions: np.ndarray, scale: Scale,
            args: argparse.Namespace) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """The predicted mean and log variance of each scan's discovered APs,
    standardised, one array per scan in the order of `positions`."""
    model.eval()
    mus, log_vars = [], []
    for start in range(0, len(positions), args.batch_size):
        scans = [cache.scans[i] for i in positions[start:start + args.batch_size]]
        batch, _, _ = make_batch(scans, scale, args, rng=None)
        with autocast(args.device):
            mu, log_var = model(batch)
        mu, log_var = mu.float().cpu().numpy(), log_var.float().cpu().numpy()
        for g, scan in enumerate(scans):
            mus.append(mu[g, :len(scan.labels)])
            log_vars.append(log_var[g, :len(scan.labels)])
    return mus, log_vars


def score(cache: Cache, test_scans: np.ndarray, mu: list[np.ndarray],
          log_var: list[np.ndarray], scale: Scale, name: str,
          objective: str = "gaussian") -> pd.DataFrame:
    """One row: every metric on the test scans, with the model's calibration.

    The model's throughput is the median of its prediction, expm1 of the
    predicted log1p mean. Coverage is the share of test APs whose log1p label
    falls inside the central 50% and 90% of their predicted distribution. The
    heuristics this is measured against are scored once, on their own, by
    baselines/train.py.

    A ranking run returns an order rather than a throughput, so the columns
    that read a prediction's scale - the regression errors and the coverage of
    its interval - are left empty for it instead of being filled with the
    reading of a number that is not a throughput.
    """
    test_frame = ap_frame(cache, test_scans)
    if objective == "ranking":
        return pd.DataFrame([{
            "model": name,
            **{column: float("nan") for column in ("mae", "rmse", "r2", "r2_log")},
            **selection_metrics(test_frame, np.concatenate(mu)),
            "coverage_50": float("nan"), "coverage_90": float("nan"),
            "mean_spread_log1p": float("nan"),
        }])
    log_mean = scale.log1p_throughput(np.concatenate(mu))
    log_spread = np.exp(0.5 * np.concatenate(log_var)) * scale.target_std
    throughput = np.clip(np.expm1(log_mean), 0.0, None)

    miss = np.abs(np.log1p(test_frame[LABEL_COL].to_numpy()) - log_mean)
    return pd.DataFrame([{
        "model": name,
        **all_metrics(test_frame, throughput),
        "coverage_50": float(np.mean(miss <= Z_50 * log_spread)),
        "coverage_90": float(np.mean(miss <= Z_90 * log_spread)),
        "mean_spread_log1p": float(np.mean(log_spread)),
    }])


def ap_frame(cache: Cache, positions: np.ndarray) -> pd.DataFrame:
    """One row per discovered AP of the given scans: the two columns common.evaluate
    scores, which are the scan it belongs to and the throughput it delivered."""
    rows = [{SCAN_COL: cache.scans[i].scan_id, LABEL_COL: float(label)}
            for i in positions for label in cache.scans[i].labels]
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--model",
                        choices=("joint_ap", "target_ap", "binned"),
                        default="joint_ap")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--only-fold", type=int, default=None,
                        help="run this one rotation instead of all of them, "
                             "leaving the deal into --folds parts unchanged")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="score the fold<k>.pt already in out_dir instead of "
                             "training; the metrics are rewritten and the "
                             "training records left as they were")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--frame-cap", type=int, default=512)
    parser.add_argument("--lr", type=float, default=None,
                        help=f"{FROM_SCRATCH_OPTIMISER.lr}, or "
                             f"{LLM_OPTIMISER.lr} for --grid-reader llm")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help=f"{FROM_SCRATCH_OPTIMISER.weight_decay}, or "
                             f"{LLM_OPTIMISER.weight_decay} for --grid-reader llm")
    parser.add_argument("--identities", type=int, default=16)
    parser.add_argument("--no-identity-tag", action="store_true",
                        help="drop the per-AP identity tag from the cell and question "
                             "vectors; for target_ap, drop the role tag instead")
    parser.add_argument("--fixed-role-codes", action="store_true",
                        help="for target_ap, use fixed orthogonal role directions "
                             "instead of learned role embeddings")
    parser.add_argument("--grid-reader", choices=("native", "llm"), default="native",
                        help="who reads the grid in stage 2: this repository's "
                             "encoder blocks, or a pretrained language model "
                             "under adapters. Ignored by --model binned")
    parser.add_argument("--objective", choices=OBJECTIVES, default="gaussian",
                        help="gaussian fits each AP's throughput and its "
                             "variance; ranking reads the head's first output "
                             "as a score and fits the choice between a scan's "
                             "APs, leaving the regression metrics undefined")
    parser.add_argument("--rank-temperature", type=float, default=1.0,
                        help="how sharp --objective ranking makes the target "
                             "distribution over a scan's APs; 0 puts all of it "
                             "on the best AP")
    parser.add_argument("--tag-mode", choices=TAG_MODES, default="codes",
                        help="fixed tag directions with learned lengths, or "
                             "learned vectors initialised at standard deviation 0.02")
    parser.add_argument("--tag-scale", type=float, default=None,
                        help="initial length of fixed tags (default 0.5)")
    parser.add_argument("--no-encoder-bias", action="store_true",
                        help="drop the learned channel-distance bias from every "
                             "encoder block")
    parser.add_argument("--no-readout-bias", action="store_true",
                        help="drop the learned channel-distance and own-AP biases "
                             "from cross-attention, leaving only the padding mask")
    parser.add_argument("--backbone", default=None,
                        help="which language model --grid-reader llm reads the grid "
                             "with; llm_model.DEFAULT_BACKBONE unless given")
    parser.add_argument("--lora-rank", type=int, default=8,
                        help="rank of the adapters on the backbone's query and "
                             "value projections")
    parser.add_argument("--llm-gradient-checkpointing", action="store_true",
                        help="recompute LLM activations during backward to reduce "
                             "GPU memory use")
    parser.add_argument("--random-backbone", action="store_true",
                        help="draw the backbone's frozen weights from its config "
                             "rather than its checkpoint: the control for whether "
                             "pretraining is what helped")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.rank_temperature < 0:
        parser.error("--rank-temperature is a temperature, so it cannot be negative")
    if args.objective == "gaussian" and args.epochs <= WARMUP_EPOCHS:
        parser.error(f"--epochs must exceed the {WARMUP_EPOCHS} warm-up epochs, "
                     "or no epoch trains the variance and none can be kept")
    if args.only_fold is not None and not 0 <= args.only_fold < args.folds:
        parser.error(f"--only-fold {args.only_fold} is not one of the "
                     f"{args.folds} folds the corpus is dealt into")
    settings = (LLM_OPTIMISER if args.grid_reader == "llm"
                else FROM_SCRATCH_OPTIMISER)
    if args.lr is None:
        args.lr = settings.lr
    if args.weight_decay is None:
        args.weight_decay = settings.weight_decay
    if args.grid_reader == "llm" and args.backbone is None:
        # resolved once here, so a checkpoint records the name it trained under
        # rather than the None that stood for it on the command line.
        from scripts.tf.llm_model import DEFAULT_BACKBONE
        args.backbone = DEFAULT_BACKBONE
    rotations = (range(args.folds) if args.only_fold is None
                 else [args.only_fold])
    args.out = args.out or Path("results") / args.model
    args.out.mkdir(parents=True, exist_ok=True)

    started = time.time()
    cache = Cache(args.cache_dir)
    print(f"{args.model}: {len(cache.scans)} scans from {args.cache_dir} in "
          f"{time.time() - started:.0f}s; device {args.device}")

    results, records = [], {}
    for fold in rotations:
        print(f"\n--- fold {fold} ---", flush=True)
        fold_results, record = run_fold(cache, fold, args)
        fold_results.insert(0, "fold", fold)
        results.append(fold_results)
        records[f"fold{fold}"] = record
        shown = fold_results.set_index("model").loc[
            [args.model], ["mae", "mean_regret_mbps", "mean_spearman"]]
        print(shown.to_string(float_format=lambda x: f"{x:.3f}"), flush=True)

    raw = pd.concat(results, ignore_index=True)
    raw.to_csv(args.out / "results_raw.csv", index=False)
    summary = raw.drop(columns="fold").groupby("model", sort=False).agg(["mean", "std"])
    summary.to_csv(args.out / "summary.csv")
    # Only a run that trained has a training record to write; scoring saved
    # models leaves the curves of the run that produced them in place.
    if not args.evaluate_only:
        (args.out / "training.json").write_text(json.dumps(records, indent=2))
    columns = ["mae", "rmse", "r2", "top1_rate", "regret_cdf_5pct",
               "mean_regret_mbps", "mean_spearman", "coverage_50", "coverage_90"]
    print("\nmean over folds:" if len(rotations) > 1
          else f"\nfold {rotations[0]}:")
    print(summary.xs("mean", axis=1, level=1)[columns].to_string(
        float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
