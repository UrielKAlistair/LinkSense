"""The grid model that scores one discovered AP per pass.

The cells are the joint model's and are encoded once per scan. What changes is
that this model's identity tag depends on the question: every AP cell is stamped
with its role relative to the AP being scored - the target itself, a candidate on
the target's channel, or a candidate on another channel. A tag that moves with
the question cannot be shared, so the grid is copied once per discovered AP and
stage 2 runs over every copy. Each copy then carries a single question, which
leaves no rivals to confer with, so rival APs reach the score only through their
cells.

That is the whole of the difference. Everything else is CellGridModel's.
The target-relative roles use learned vectors by default, while the cell type,
channel and dwell tags follow the shared fixed-tag embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from scripts.tf.cell_grid_model import CellGridModel
from scripts.tf.layers import CellBatch, CodeBook

# The role an AP cell takes relative to the AP being scored.
TARGET_AP, CO_CHANNEL_AP, OTHER_AP = 0, 1, 2


class TargetAPModel(CellGridModel):
    """Scores one discovered AP per pass; a batch runs one pass per discovered AP."""

    # One question per grid copy, so there is nothing to confer with.
    compare = False

    def __init__(self, n_frame_features: int, n_aggregates: int, n_descriptor: int,
                 role_tag: bool = True, fixed_role_codes: bool = False, **kwargs):
        # Set before the base constructor, which calls make_identity_tables while
        # it is still running.
        self.role_tag = role_tag
        self.fixed_role_codes = fixed_role_codes
        if fixed_role_codes and kwargs.get("tag_mode", "codes") != "codes":
            raise ValueError("fixed_role_codes requires fixed common tags")
        super().__init__(n_frame_features, n_aggregates, n_descriptor, **kwargs)

    def identity_size(self) -> int:
        return 3 if self.role_tag else 0

    def make_identity_tags(self, tags, length: float) -> None:
        if not self.role_tag:
            self.role = None
        elif self.fixed_role_codes:
            self.role = CodeBook(tags, length)
        else:
            self.role = nn.Embedding(3, tags.shape[1])
            nn.init.normal_(self.role.weight, std=0.02)

    def make_identity_tables(self, tag_width: int) -> list[nn.Embedding]:
        if not self.role_tag:
            self.role = None
            return []
        self.role = nn.Embedding(3, tag_width)
        return [self.role]

    def forward(self, batch: CellBatch):
        content, tags = self.cell_parts(batch)

        # One copy of the grid per question: copy p asks about AP ap[p] of scan
        # scan[p], and carries that AP's roles stamped onto its cells.
        scan, ap = batch.ap_mask.nonzero(as_tuple=True)
        target_channel = batch.ap_channel[scan, ap][:, None]
        target_identity = batch.ap_identity[scan, ap][:, None]
        is_channel, channel = batch.cell_is_channel[scan], batch.cell_channel[scan]
        identity, cell_mask = batch.cell_ap_identity[scan], batch.cell_mask[scan]
        content, tags = content[scan], tags[scan]
        if self.role_tag:
            is_ap = (~is_channel).unsqueeze(-1)
            tags = tags + is_ap * self.role(
                roles(target_channel, target_identity, channel, identity))
        grid = self.read_grid(self.join(content, tags),
                              batch.cell_dwell[scan], channel, cell_mask)

        rows = self.build_rows(batch.descriptors[scan, ap].unsqueeze(1),
                               target_channel, None)
        bias = self.readout_mask(target_channel, target_identity, channel,
                                 is_channel, identity, cell_mask, grid.dtype)
        mu, log_var = self.score(rows, grid, None, bias)

        # Back into the (scans, APs) layout the joint model produces directly.
        out = mu.new_zeros(*batch.ap_mask.shape, 2)
        out[scan, ap] = torch.stack([mu.squeeze(1), log_var.squeeze(1)], dim=-1)
        return out[..., 0], out[..., 1]


def roles(target_channel: torch.Tensor, target_identity: torch.Tensor,
          cell_channel: torch.Tensor, cell_ap_identity: torch.Tensor) -> torch.Tensor:
    """Each cell's role relative to the target, read as if it were an AP cell;
    the model adds it to AP cells only."""
    return torch.where(cell_ap_identity == target_identity, TARGET_AP,
                       torch.where(cell_channel == target_channel, CO_CHANNEL_AP, OTHER_AP))
