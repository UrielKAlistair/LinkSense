"""The grid model that scores every discovered AP of a scan in one pass.

One question row per discovered AP cross-attends into a single grid, and the
rows also attend to each other, which is what makes this joint rather than a
per-AP model run once per AP: a row reads its rivals' rows as well as their
cells. Stage 2 therefore runs once per scan.

Identity stays relational. An AP token carries an identity embedding drawn at
random per sample, shared with its question row, so a row can find its own AP's
cells without the model ever seeing an address it could memorise; a learned bias
on cross-attention also marks those cells directly. That tag does not depend on
which AP is being scored, which is what lets one grid serve every question.

Everything else is CellGridModel's, and target_ap_model.py builds from it too.
"""

from __future__ import annotations

import torch.nn as nn

from scripts.tf.cell_grid_model import CellGridModel
from scripts.tf.layers import CellBatch, CodeBook


class JointAPModel(CellGridModel):
    """Scores every discovered AP of a scan in one pass."""

    # The rows are rivals within one scan, so they confer before reading the grid.
    compare = True

    def __init__(self, n_frame_features: int, n_aggregates: int, n_descriptor: int,
                 n_identities: int = 16, ap_identity_tag: bool = True, **kwargs):
        # Set before the base constructor, which calls make_identity_tables while
        # it is still running so that every tag table is drawn from one point in
        # the seeded stream.
        self.n_identities = n_identities
        # An ablation switch. Off, identity leaves the token entirely and reaches
        # attention only through the readout bias, or not at all.
        self.ap_identity_tag = ap_identity_tag
        super().__init__(n_frame_features, n_aggregates, n_descriptor, **kwargs)

    def identity_size(self) -> int:
        return self.n_identities if self.ap_identity_tag else 0

    def make_identity_tags(self, tags, length: float) -> None:
        self.ap_identity = CodeBook(tags, length) if self.ap_identity_tag else None

    def make_identity_tables(self, tag_width: int) -> list[nn.Embedding]:
        if not self.ap_identity_tag:
            self.ap_identity = None
            return []
        self.ap_identity = nn.Embedding(self.n_identities, tag_width)
        return [self.ap_identity]

    def cell_identity(self, batch: CellBatch):
        """Which AP a cell belongs to, on AP cells alone."""
        if not self.ap_identity_tag:
            return None
        is_ap = (~batch.cell_is_channel).unsqueeze(-1)
        return is_ap * self.ap_identity(batch.cell_ap_identity)

    def forward(self, batch: CellBatch):
        grid = self.read_grid(self.encode_cells(batch), batch.cell_dwell,
                              batch.cell_channel, batch.cell_mask)

        # A row's tags go in the same dimensions a cell's do, so that the two meet
        # in the bilinear form attention scores them with.
        tags = self.ap_identity(batch.ap_identity) if self.ap_identity_tag else None
        rows = self.build_rows(batch.descriptors, batch.ap_channel, tags)
        bias = self.readout_mask(batch.ap_channel, batch.ap_identity,
                                 batch.cell_channel, batch.cell_is_channel,
                                 batch.cell_ap_identity, batch.cell_mask, grid.dtype)
        return self.score(rows, grid, ~batch.ap_mask, bias)
