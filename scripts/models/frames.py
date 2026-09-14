"""A transformer over a raw 802.11 frame trace, with no time binning at all.

temporal.py compresses the scan into fixed time slices before the network sees
it, which decides in advance what time resolution matters. This module does
not: the input is the list of decoded frames as they arrived, and the only
aggregation is the one attention learns.

READ THIS BEFORE TREATING THIS AS A SEQUENCE MODEL. There is no positional
encoding over the frame axis and cross-attention is permutation invariant, so
permuting the frames leaves every score unchanged. Time reaches the model only
as two ordinary feature columns - `time_fraction` and `gap_since_previous`,
both set in build_frame_corpus.py - which the network sees exactly as it sees
signal strength. This is a bag of timestamped frames, not a trajectory, and a
shuffled-order ablation against it is a no-op by construction.

Two constraints shape the architecture.

Frames are many and options are few. A rotating radio decodes thousands of
frames over a scan, so self-attention across the trace would be quadratic in
the wrong quantity. Instead each option carries a small set of learned query
vectors that cross-attend INTO the trace: cost is linear in the number of
frames, and only the per-option summaries talk to each other afterwards.

Identity must be relational. Frames carry no BSSID here. Each frame is tagged,
per option, with a code saying only how it relates to the option currently
asking: was it on that option's channel, from that option's BSS, from that AP
itself. One shared trace is therefore read differently by each option without
the model ever seeing an address it could memorise, so it cannot learn that a
particular AP is usually fast and must work from what the frames sound like.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import split_rows_by_topology

FRAME_SCHEMA_VERSION = 3

# How one frame relates to one option, as independent bits combined into a
# single code per (frame, option) pair. scripts/dataset/build_frame_corpus.py
# sets these bits and FrameSetTransformer embeds the result, so a change here
# invalidates every frame corpus already built.
REL_ON_CHANNEL = 1      # the frame was on the channel this option's AP uses
REL_SAME_BSS = 2        # the frame came from a device in this option's BSS
REL_FROM_AP = 4         # the frame was sent by this option's AP itself
# The bits are nested, not independent: a frame from this option's AP is also
# from its BSS and on its channel, so only codes 0, 1, 3 and 7 ever occur.
# 8 is still the right indexing bound - four of its rows simply stay unused.
N_RELATIONS = 8


@dataclass
class FrameCorpus:
    """One decoded frame trace per group, with the options it chose between.

    A group's trace is shared by all of its options; `relations` is what makes
    each option read that one trace differently.
    """

    frames: np.ndarray          # (G, N, F) per-frame measurements
    relations: np.ndarray       # (G, O, N) uint8, one relation code per option
    frame_mask: np.ndarray      # (G, N) which frame slots are real
    labels: np.ndarray          # (G, O) measured throughput
    option_indices: np.ndarray
    option_mask: np.ndarray     # (G, O) which option slots are real
    scan_ids: np.ndarray
    topology_ids: np.ndarray
    configured_n_aps: np.ndarray
    n_hotspots: np.ndarray
    window_s: np.ndarray        # (G,) listening window length in seconds
    frame_features: list[str]

    @classmethod
    def load(cls, path: Path) -> "FrameCorpus":
        with np.load(path, allow_pickle=False) as d:
            if int(d["schema_version"]) != FRAME_SCHEMA_VERSION:
                raise ValueError(
                    f"frame corpus at {path} is schema version "
                    f"{int(d['schema_version'])}, expected {FRAME_SCHEMA_VERSION}; "
                    "rebuild it with scripts/dataset/build_frame_corpus.py")
            return cls(
                frames=d["frames"], relations=d["relations"],
                frame_mask=d["frame_mask"], labels=d["labels"],
                option_indices=d["option_indices"], option_mask=d["option_mask"],
                scan_ids=d["scan_ids"], topology_ids=d["topology_ids"],
                configured_n_aps=d["configured_n_aps"], n_hotspots=d["n_hotspots"],
                window_s=d["window_s"],
                frame_features=d["frame_features"].tolist())

    def split(self, seed: int = 0, val_frac: float = 0.2, test_frac: float = 0.2):
        """Group positions for train, val and test, split by whole topology.

        Every repeated seed of a deployment stays on one side of the split.
        Balanced across the number of APs a deployment was configured with, so
        no part is short of the large or the small deployments. Identical rule
        to TemporalCorpus.split, so a frame model and a binned model trained
        with the same seed are scored on the same deployments.
        """
        return split_rows_by_topology(self.topology_ids, self.configured_n_aps,
                                      val_frac, test_frac, seed)


class FrameSetTransformer(nn.Module):
    """Summarise a shared frame trace once per option, then compare the options.

    Each option's learned queries cross-attend into the trace tagged with that
    option's relation codes, and the pooled results attend to one another with
    no position encoding, so the scores are equivariant to the order the
    options are listed in.
    """

    def __init__(self, n_frame_features: int, model_dim: int = 48,
                 heads: int = 4, latents: int = 4,
                 cross_layers: int = 2, set_layers: int = 2, dropout: float = 0.1):
        super().__init__()
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
                frame_mask: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
        batch, n_options = option_mask.shape
        n_frames = frames.shape[1]

        base = self.frame_in(frames)                                  # (B, N, D)
        # each option sees the same frames tagged with its own relation code
        tokens = base[:, None] + self.relation(relations.long())      # (B, O, N, D)
        tokens = tokens.reshape(batch * n_options, n_frames, self.model_dim)
        padding = ~frame_mask[:, None, :].expand(batch, n_options, n_frames)
        padding = padding.reshape(batch * n_options, n_frames)
        # A scan with no frames at all leaves every key masked. Let those rows
        # attend to slot 0, whose token is zeroed padding, so the encoder sees
        # a defined query rather than an empty one.
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
        compared = self.options(pooled, src_key_padding_mask=~option_mask)
        return self.score(self.norm(compared)).squeeze(-1)

