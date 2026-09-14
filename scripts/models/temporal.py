"""A transformer over a passive scan that has been cut into time bins.

Before a client joins a network it can only listen. Sweeping its radio across
the channels for a few seconds, it hears beacons and data frames from the
access points in range. The corpus in this module is that listening period
chopped into equal time bins, with one row of measurements per bin per access
point the client could join: how strong that AP sounded in this bin, how much
of the bin its channel was occupied, whether it was heard at all. The job is
to score the options so the one that would actually deliver the most
throughput comes top.

The model reads that in two stages, which is the whole idea:

  time     each option's own sequence of bins is summarised independently by
           a transformer over the time axis, ending in one vector per option.
           Padding bins are masked, so a short scan is not read as a quiet one.
  options  those per-option vectors then attend to each other, with no
           position encoding, so the score an option receives depends on which
           rivals it is up against but not on the order they arrive in.

The second stage is what a model scoring each option in isolation cannot do:
express that an AP is only lightly loaded *compared with the others here*.

Splitting a corpus is delegated to data.split_rows_by_topology, so an array
corpus and the flat CSV are split by exactly the same rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import split_rows_by_topology

TEMPORAL_SCHEMA_VERSION = 4


@dataclass
class TemporalCorpus:
    """One binned scan per row, with the options it was choosing between.

    Leading axes are (scan, option, time step, feature); option_mask says
    which option slots are real APs rather than padding, and time_mask says
    which time steps are real bins.
    """

    temporal: np.ndarray
    labels: np.ndarray
    option_indices: np.ndarray
    option_mask: np.ndarray
    time_mask: np.ndarray
    scan_ids: np.ndarray
    topology_ids: np.ndarray
    configured_n_aps: np.ndarray
    n_hotspots: np.ndarray
    temporal_features: list[str]

    @classmethod
    def load(cls, path: Path) -> "TemporalCorpus":
        with np.load(path, allow_pickle=False) as data:
            if int(data["schema_version"]) != TEMPORAL_SCHEMA_VERSION:
                raise ValueError(
                    f"temporal corpus at {path} is schema version "
                    f"{int(data['schema_version'])}, expected "
                    f"{TEMPORAL_SCHEMA_VERSION}; rebuild it with "
                    "scripts/dataset/build_binned_corpus.py")
            return cls(
                temporal=data["temporal"], labels=data["labels"],
                option_indices=data["option_indices"], option_mask=data["option_mask"],
                time_mask=data["time_mask"], scan_ids=data["scan_ids"],
                topology_ids=data["topology_ids"],
                configured_n_aps=data["configured_n_aps"],
                n_hotspots=data["n_hotspots"],
                temporal_features=data["temporal_features"].tolist(),
            )

    def split(self, seed: int = 0, val_frac: float = 0.2,
              test_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Group positions for train, val and test, split by whole topology.

        Every repeated seed of a deployment stays on one side of the split.
        Balanced across the number of APs a deployment was configured with, so
        no part is short of the large or the small deployments.
        """
        return split_rows_by_topology(self.topology_ids, self.configured_n_aps,
                                      val_frac, test_frac, seed)


class MaskedStandardizer:
    """Feature scaling whose mean and variance ignore padded slots.

    Fitted on training data only. `mask` selects the entries of `values` that
    are real observations; including padding would drag every statistic toward
    whatever the padding happens to hold. A feature with no spread is still
    centred, but its scale is left at 1.0 rather than dividing by nearly zero.
    """

    def __init__(self, values: np.ndarray, mask: np.ndarray):
        valid = values[mask]
        self.mean = valid.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.std = valid.std(axis=0, dtype=np.float64).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0

    def __call__(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)


class TemporalSetTransformer(nn.Module):
    """Encode time per AP, then compare AP embeddings without option order."""

    def __init__(self, n_temporal_features: int, max_steps: int,
                 model_dim: int = 48, heads: int = 4, temporal_layers: int = 2,
                 set_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.max_steps = max_steps
        self.input = nn.Linear(n_temporal_features, model_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.position = nn.Parameter(torch.zeros(1, max_steps + 1, model_dim))

        temporal_layer = nn.TransformerEncoderLayer(
            model_dim, heads, dim_feedforward=model_dim * 3, dropout=dropout,
            batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(
            temporal_layer, temporal_layers, enable_nested_tensor=False)
        set_layer = nn.TransformerEncoderLayer(
            model_dim, heads, dim_feedforward=model_dim * 3, dropout=dropout,
            batch_first=True, norm_first=True)
        self.options = nn.TransformerEncoder(
            set_layer, set_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(model_dim)
        self.score = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, temporal: torch.Tensor, option_mask: torch.Tensor,
                time_mask: torch.Tensor) -> torch.Tensor:
        batch, options, steps, _ = temporal.shape
        if steps > self.max_steps:
            raise ValueError(
                f"input has {steps} time steps but this model was built with "
                f"max_steps={self.max_steps}; it has no position encoding for "
                "the extra steps")
        tokens = self.input(temporal).reshape(batch * options, steps, -1)
        cls = self.cls.expand(batch * options, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.position[:, :steps + 1]

        # One time mask per scan: it marks which BINS EXIST, not which options
        # were heard in them. A bin where this option was silent still carries a
        # level - forward-filled by the builder, with option_age saying how
        # stale it is - so masking per option here would discard those.
        temporal_padding = ~time_mask[:, None, :].expand(batch, options, steps)
        temporal_padding = temporal_padding.reshape(batch * options, steps)
        # The prepended CLS token is never padding; it is what carries the
        # summary of the sequence out of this stage.
        temporal_padding = torch.cat([
            torch.zeros((batch * options, 1), dtype=torch.bool, device=temporal.device),
            temporal_padding,
        ], dim=1)
        encoded = self.temporal(tokens, src_key_padding_mask=temporal_padding)
        option_embeddings = encoded[:, 0].reshape(batch, options, -1)

        compared = self.options(option_embeddings, src_key_padding_mask=~option_mask)
        return self.score(self.norm(compared)).squeeze(-1)


def load_temporal_checkpoint(path: Path, map_location: str | torch.device = "cpu"
                             ) -> tuple[TemporalSetTransformer, dict]:
    """Rebuild a trained model from a file written by scripts/train/train_temporal.py.

    The checkpoint's "model" entry holds the constructor arguments, so this is
    the only place that knows how to turn a saved .pt back into a working
    model; without it the saved runs can only be reopened by working out those
    arguments by hand. Loaded with weights_only so opening a checkpoint cannot
    execute code from it.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model = TemporalSetTransformer(**checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
