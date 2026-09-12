#!/usr/bin/env python3
"""Merge independently generated sweeps into one dataset.

Sweeps run at different times, or on different machines, produce separate
tables over disjoint topologies. This joins them without reshaping anything:
schemas must already match, and topology IDs must not collide, since a
repeated ID would put the same physical deployment on both sides of a split.

Padded NPZ corpora may differ in their option and time dimensions. Valid
values are copied into the largest required shape, so the padding grows and
no real value moves.

This never splits anything. Train/validation/test partitioning happens at
training time, by topology, in models/data.py.

Run:
  python scripts/dataset/combine_datasets.py data/a.csv data/b.csv --out data/all.csv
  python scripts/dataset/combine_datasets.py data/a.npz data/b.npz --out data/all.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def combine_csv(paths: list[Path], out: Path) -> None:
    frames = [pd.read_csv(path) for path in paths]
    expected = list(frames[0].columns)
    for path, frame in zip(paths[1:], frames[1:]):
        if list(frame.columns) != expected:
            raise ValueError(f"{path}: columns or column order do not match the first input")

    combined = pd.concat(frames, ignore_index=True)

    # A topology appearing in two inputs would be split across train and test.
    seen: set = set()
    overlap: set = set()
    for frame in frames:
        current = set(frame["topology_id"])
        overlap |= seen & current
        seen |= current
    if overlap:
        raise ValueError(f"duplicate topology IDs across inputs: {sorted(overlap)[:10]}")

    # Scan and run IDs must stay nested inside one topology, or splitting by
    # topology no longer separates the scans.
    for column in ("scan_id", "run_id"):
        if column not in combined:
            continue
        owners = combined.groupby(column, dropna=False)["topology_id"].nunique()
        if (owners > 1).any():
            raise ValueError(f"{column} maps to more than one topology")

    combined = combined.sort_values(["topology_id", "scan_id", "ap_index"])
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    print(f"wrote {len(combined)} rows, {combined.scan_id.nunique()} scans, "
          f"{combined.topology_id.nunique()} topologies to {out}")


def _constant_equal(first: np.ndarray, other: np.ndarray) -> bool:
    return first.shape == other.shape and np.array_equal(first, other)


def combine_npz(paths: list[Path], out: Path) -> None:
    corpora = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            corpora.append({key: data[key] for key in data.files})

    constant_keys = ("schema_version", "temporal_features", "static_features",
                     "bin_ms", "scan_description")
    for path, corpus in zip(paths[1:], corpora[1:]):
        mismatched = [key for key in constant_keys
                      if not _constant_equal(corpora[0][key], corpus[key])]
        if mismatched:
            raise ValueError(f"{path}: temporal schema mismatch in {mismatched}")

    topology_sets = [set(corpus["topology_ids"].tolist()) for corpus in corpora]
    seen: set[str] = set()
    for path, current in zip(paths, topology_sets):
        overlap = seen & current
        if overlap:
            raise ValueError(f"{path}: duplicate topology IDs {sorted(overlap)[:10]}")
        seen |= current

    max_options = max(corpus["temporal"].shape[1] for corpus in corpora)
    max_steps = max(corpus["temporal"].shape[2] for corpus in corpora)
    n_temporal = corpora[0]["temporal"].shape[-1]
    n_static = corpora[0]["static"].shape[-1]

    array_keys = ("temporal", "static", "labels", "option_indices",
                  "option_mask", "time_mask")
    padded = {key: [] for key in array_keys}
    for corpus in corpora:
        scans, options, steps, _ = corpus["temporal"].shape
        temporal = np.zeros((scans, max_options, max_steps, n_temporal), np.float32)
        static = np.zeros((scans, max_options, n_static), np.float32)
        labels = np.zeros((scans, max_options), np.float32)
        indices = np.full((scans, max_options), -1, np.int16)
        option_mask = np.zeros((scans, max_options), bool)
        time_mask = np.zeros((scans, max_steps), bool)

        temporal[:, :options, :steps] = corpus["temporal"]
        static[:, :options] = corpus["static"]
        labels[:, :options] = corpus["labels"]
        indices[:, :options] = corpus["option_indices"]
        option_mask[:, :options] = corpus["option_mask"]
        time_mask[:, :steps] = corpus["time_mask"]
        for key, value in zip(array_keys, (temporal, static, labels, indices,
                                           option_mask, time_mask)):
            padded[key].append(value)

    vector_keys = ("scan_ids", "topology_ids", "configured_n_aps",
                   "n_hotspots", "candidate_strata")
    payload = {key: np.concatenate(values) for key, values in padded.items()}
    payload.update({key: np.concatenate([corpus[key] for corpus in corpora])
                    for key in vector_keys})
    payload.update({key: corpora[0][key] for key in constant_keys})

    if len(set(payload["scan_ids"].tolist())) != len(payload["scan_ids"]):
        raise ValueError("duplicate scan IDs across inputs")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    print(f"wrote {len(payload['scan_ids'])} scans, "
          f"{int(payload['option_mask'].sum())} options, {len(seen)} topologies to {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    suffixes = {path.suffix.lower() for path in args.inputs}
    if len(args.inputs) < 2:
        parser.error("at least two inputs are required")
    if suffixes == {".csv"} and args.out.suffix.lower() == ".csv":
        combine_csv(args.inputs, args.out)
    elif suffixes == {".npz"} and args.out.suffix.lower() == ".npz":
        combine_npz(args.inputs, args.out)
    else:
        parser.error("inputs and --out must all be CSV or all be NPZ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
