"""Perceiver-style transformer over a raw 802.11 frame trace.

The binned models compress the scan into fixed time slices before the network
sees it. This one does not: the input is the frame list, and the only
aggregation is the one attention learns.

Two constraints shape the architecture.

Frames are many and options are few. A rotating radio over a 5.5 s window
decodes thousands of frames, so self-attention over the trace is quadratic in
the wrong quantity. Instead a small set of learned latents per option
cross-attends INTO the trace: cost is linear in the number of frames, and the
option latents are the only things that talk to each other afterwards.

Identity must be relational. Frames carry no BSSID here - only a 3-bit code
saying how each frame relates to the option currently asking (its channel, its
BSS, its AP). One shared trace is therefore read differently by each option
without the model ever seeing an address it could memorise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import partition_groups

N_RELATIONS = 8


@dataclass
class FrameCorpus:
    frames: np.ndarray          # (G, N, F)
    relations: np.ndarray       # (G, O, N) uint8
    frame_mask: np.ndarray      # (G, N)
    static: np.ndarray          # (G, O, S)
    labels: np.ndarray          # (G, O)
    option_indices: np.ndarray
    option_mask: np.ndarray     # (G, O)
    group_ids: np.ndarray
    topology_ids: np.ndarray
    configured_n_aps: np.ndarray
    n_hotspots: np.ndarray
    candidate_strata: np.ndarray
    frame_features: list[str]
    static_features: list[str]

    @classmethod
    def load(cls, path: Path) -> "FrameCorpus":
        with np.load(path, allow_pickle=False) as d:
            if int(d["schema_version"]) != 1:
                raise ValueError("unsupported frame corpus schema")
            return cls(
                frames=d["frames"], relations=d["relations"],
                frame_mask=d["frame_mask"], static=d["static"], labels=d["labels"],
                option_indices=d["option_indices"], option_mask=d["option_mask"],
                group_ids=d["group_ids"], topology_ids=d["topology_ids"],
                configured_n_aps=d["configured_n_aps"], n_hotspots=d["n_hotspots"],
                candidate_strata=d["candidate_strata"],
                frame_features=d["frame_features"].tolist(),
                static_features=d["static_features"].tolist())

    def split(self, seed: int = 0, val_frac: float = 0.2, test_frac: float = 0.2):
        topologies = np.unique(self.topology_ids)
        if len(topologies) < 5:
            raise ValueError("at least 5 independent topologies are required")
        strata = np.array([
            self.configured_n_aps[np.flatnonzero(self.topology_ids == t)[0]]
            for t in topologies])
        chosen = tuple(set(part) for part in
                       partition_groups(topologies, strata, val_frac, test_frac, seed))
        return tuple(np.flatnonzero(np.isin(self.topology_ids, list(names)))
                     for names in chosen)


class FrameStandardizer:
    """Fitted on real frames only; padding must not move the statistics."""

    def __init__(self, frames: np.ndarray, mask: np.ndarray):
        valid = frames[mask]
        self.mean = valid.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.std = valid.std(axis=0, dtype=np.float64).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        return ((frames - self.mean) / self.std).astype(np.float32)


class FrameSetTransformer(nn.Module):
    def __init__(self, n_frame_features: int, n_static_features: int = 0,
                 model_dim: int = 48, heads: int = 4, latents: int = 4,
                 cross_layers: int = 2, set_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.latents = latents
        self.model_dim = model_dim
        self.frame_in = nn.Linear(n_frame_features, model_dim)
        # the only identity the model ever sees
        self.relation = nn.Embedding(N_RELATIONS, model_dim)
        nn.init.normal_(self.relation.weight, std=0.02)
        self.query = nn.Parameter(torch.zeros(1, latents, model_dim))
        nn.init.normal_(self.query, std=0.02)

        self.cross = nn.ModuleList([
            nn.MultiheadAttention(model_dim, heads, dropout=dropout, batch_first=True)
            for _ in range(cross_layers)])
        self.cross_norm_q = nn.ModuleList([nn.LayerNorm(model_dim) for _ in range(cross_layers)])
        self.cross_norm_kv = nn.ModuleList([nn.LayerNorm(model_dim) for _ in range(cross_layers)])
        self.cross_ff = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, model_dim * 3),
                          nn.GELU(), nn.Dropout(dropout), nn.Linear(model_dim * 3, model_dim))
            for _ in range(cross_layers)])

        self.static = (nn.Sequential(nn.Linear(n_static_features, model_dim),
                                     nn.GELU(), nn.LayerNorm(model_dim))
                       if n_static_features else None)
        set_layer = nn.TransformerEncoderLayer(
            model_dim, heads, dim_feedforward=model_dim * 3, dropout=dropout,
            batch_first=True, norm_first=True)
        self.options = nn.TransformerEncoder(set_layer, set_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(model_dim)
        self.score = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(model_dim, 1))

    def forward(self, frames: torch.Tensor, relations: torch.Tensor,
                frame_mask: torch.Tensor, option_mask: torch.Tensor,
                static: torch.Tensor | None = None) -> torch.Tensor:
        batch, n_options = option_mask.shape
        n_frames = frames.shape[1]

        base = self.frame_in(frames)                                  # (B, N, D)
        # each option sees the same frames tagged with its own relation code
        tokens = base[:, None] + self.relation(relations.long())      # (B, O, N, D)
        tokens = tokens.reshape(batch * n_options, n_frames, self.model_dim)
        padding = ~frame_mask[:, None, :].expand(batch, n_options, n_frames)
        padding = padding.reshape(batch * n_options, n_frames)
        # A group with no frames at all would make every key invalid, which is
        # undefined for softmax attention; let such rows attend to slot 0 and
        # rely on the option mask to discard them downstream.
        empty = padding.all(dim=1)
        if empty.any():
            padding[empty, 0] = False

        q = self.query.expand(batch * n_options, -1, -1)
        for attention, norm_q, norm_kv, ff in zip(
                self.cross, self.cross_norm_q, self.cross_norm_kv, self.cross_ff):
            kv = norm_kv(tokens)
            attended, _ = attention(norm_q(q), kv, kv, key_padding_mask=padding,
                                    need_weights=False)
            q = q + attended
            q = q + ff(q)

        pooled = q.mean(dim=1).reshape(batch, n_options, self.model_dim)
        if self.static is not None:
            if static is None:
                raise ValueError("static option features are required by this model")
            pooled = pooled + self.static(static)
        compared = self.options(pooled, src_key_padding_mask=~option_mask)
        return self.score(self.norm(compared)).squeeze(-1)
