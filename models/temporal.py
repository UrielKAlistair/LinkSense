"""Temporal and cross-option transformer for passive Wi-Fi scans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import partition_groups


@dataclass
class TemporalCorpus:
    temporal: np.ndarray
    static: np.ndarray
    labels: np.ndarray
    option_indices: np.ndarray
    option_mask: np.ndarray
    time_mask: np.ndarray
    group_ids: np.ndarray
    topology_ids: np.ndarray
    configured_n_aps: np.ndarray
    n_hotspots: np.ndarray
    candidate_strata: np.ndarray
    temporal_features: list[str]
    static_features: list[str]

    @classmethod
    def load(cls, path: Path) -> "TemporalCorpus":
        with np.load(path, allow_pickle=False) as data:
            if int(data["schema_version"]) != 2:
                raise ValueError("unsupported temporal dataset schema")
            return cls(
                temporal=data["temporal"], static=data["static"], labels=data["labels"],
                option_indices=data["option_indices"], option_mask=data["option_mask"],
                time_mask=data["time_mask"], group_ids=data["group_ids"],
                topology_ids=data["topology_ids"],
                configured_n_aps=data["configured_n_aps"],
                n_hotspots=data["n_hotspots"],
                candidate_strata=data["candidate_strata"],
                temporal_features=data["temporal_features"].tolist(),
                static_features=data["static_features"].tolist(),
            )

    def split(self, seed: int = 0, val_frac: float = 0.2,
              test_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split whole physical topologies, keeping their five seeds together."""
        topologies = np.unique(self.topology_ids)
        if len(topologies) < 5:
            raise ValueError("at least 5 independent topologies are required for splitting")
        topology_strata = np.array([
            self.configured_n_aps[np.flatnonzero(self.topology_ids == topology)[0]]
            for topology in topologies
        ])
        selected = tuple(set(part) for part in partition_groups(
            topologies, topology_strata, val_frac, test_frac, seed))
        return tuple(np.flatnonzero(np.isin(self.topology_ids, list(names)))
                     for names in selected)


class MaskedStandardizer:
    """Feature scaling fitted only on valid training tokens and options."""

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
                 n_static_features: int = 0, model_dim: int = 48,
                 heads: int = 4, temporal_layers: int = 2, set_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.n_static_features = n_static_features
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
        self.static = (nn.Sequential(
            nn.Linear(n_static_features, model_dim), nn.GELU(), nn.LayerNorm(model_dim))
            if n_static_features else None)
        self.norm = nn.LayerNorm(model_dim)
        self.score = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, temporal: torch.Tensor, option_mask: torch.Tensor,
                time_mask: torch.Tensor,
                static: torch.Tensor | None = None) -> torch.Tensor:
        batch, options, steps, _ = temporal.shape
        tokens = self.input(temporal).reshape(batch * options, steps, -1)
        cls = self.cls.expand(batch * options, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.position[:, :steps + 1]

        temporal_padding = ~time_mask[:, None, :].expand(batch, options, steps)
        temporal_padding = temporal_padding.reshape(batch * options, steps)
        temporal_padding = torch.cat([
            torch.zeros((batch * options, 1), dtype=torch.bool, device=temporal.device),
            temporal_padding,
        ], dim=1)
        encoded = self.temporal(tokens, src_key_padding_mask=temporal_padding)
        option_embeddings = encoded[:, 0].reshape(batch, options, -1)
        if self.static is not None:
            if static is None:
                raise ValueError("static option features are required by this model")
            option_embeddings = option_embeddings + self.static(static)

        compared = self.options(option_embeddings, src_key_padding_mask=~option_mask)
        return self.score(self.norm(compared)).squeeze(-1)


def load_temporal_checkpoint(path: Path, map_location: str | torch.device = "cpu"
                             ) -> tuple[TemporalSetTransformer, dict]:
    """Safely reconstruct a temporal model and return its inference metadata."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model = TemporalSetTransformer(**checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
