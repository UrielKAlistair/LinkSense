"""The binned transformer, reading the cell cache.

This model reads each AP's view of the scan as a sequence of time bins, one bin
per dwell. A bin is a pair of cells, so it takes the same batches as the cell
models and lays each AP's sequence out from them. At every dwell an AP sees:

  the channel cell's aggregates    everything heard on the tuned channel
  its own cell's aggregates        zero on dwells on other channels
  whether the dwell is on its channel
  the share of the scan's discovered APs on the tuned channel

No frame, carrier-sense trace or window descriptor reaches it.

The model reads that in two stages:

  time     each AP's dwells are read alone, by self-attention with a learned
           position per dwell, and summarised by a prepended summary token.
  APs      the summaries attend to each other, with no order, so an AP is
           scored against the rivals it is up against.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.tf.layers import LOG_VAR_RANGE, CellBatch


class BinnedModel(nn.Module):
    """Reads each discovered AP's dwells in time, then compares the APs."""

    def __init__(self, n_aggregates: int, n_dwells: int = 52, width: int = 48,
                 heads: int = 4, time_layers: int = 2, set_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.n_dwells = n_dwells
        self.input = nn.Linear(2 * n_aggregates + 2, width)
        self.summary = nn.Parameter(torch.zeros(1, 1, width))
        self.position = nn.Parameter(torch.zeros(1, n_dwells + 1, width))
        nn.init.normal_(self.summary, std=0.02)
        nn.init.normal_(self.position, std=0.02)

        def layer() -> nn.TransformerEncoderLayer:
            return nn.TransformerEncoderLayer(width, heads, dim_feedforward=width * 3,
                                              dropout=dropout, batch_first=True,
                                              norm_first=True)

        self.time = nn.TransformerEncoder(layer(), time_layers, enable_nested_tensor=False)
        self.compare = nn.TransformerEncoder(layer(), set_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, 2))

    def forward(self, batch: CellBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and log variance of the standardised log1p throughput, one pair
        per discovered AP."""
        steps, dwell_exists = binned_view(batch, self.n_dwells)

        # 1. time, once per discovered AP: pair i is AP ap[i] of scan scan[i]
        scan, ap = batch.ap_mask.nonzero(as_tuple=True)
        tokens = self.input(steps[scan, ap])
        tokens = torch.cat([self.summary.expand(len(scan), -1, -1), tokens], dim=1)
        tokens = tokens + self.position
        # the summary token is never padding
        padding = F.pad(~dwell_exists[scan], (1, 0))
        read = self.time(tokens, src_key_padding_mask=padding)[:, 0]
        summaries = read.new_zeros(*batch.ap_mask.shape, read.shape[-1])
        summaries[scan, ap] = read

        # 2. APs
        compared = self.compare(summaries, src_key_padding_mask=~batch.ap_mask)
        prediction = self.head(self.norm(compared))
        return prediction[..., 0], prediction[..., 1].clamp(*LOG_VAR_RANGE)


def binned_view(batch: CellBatch, n_dwells: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Each AP's view of every dwell, (scans, APs, dwells, 2K + 2), and which
    dwells exist, (scans, dwells).

    Each dwell has one channel cell, and each AP one cell per dwell on its
    channel; a cell belongs to the AP whose identity it carries.
    """
    scans, aps = batch.ap_mask.shape
    width = batch.cell_aggregates.shape[-1]

    # What the radio heard on the tuned channel, and which channel that was,
    # laid out by dwell. A dwell with no channel cell never happened.
    scan, cell = (batch.cell_is_channel & batch.cell_mask).nonzero(as_tuple=True)
    dwell = batch.cell_dwell[scan, cell]
    heard = batch.cell_aggregates.new_zeros(scans, n_dwells, width)
    heard[scan, dwell] = batch.cell_aggregates[scan, cell]
    tuned = batch.cell_channel.new_zeros(scans, n_dwells)
    tuned[scan, dwell] = batch.cell_channel[scan, cell]
    dwell_exists = torch.zeros(scans, n_dwells, dtype=torch.bool, device=tuned.device)
    dwell_exists[scan, dwell] = True

    # Each AP's own cells, matched by the identity they carry, in that same layout.
    owner = (((~batch.cell_is_channel) & batch.cell_mask)[:, :, None]
             & (batch.cell_ap_identity[:, :, None] == batch.ap_identity[:, None, :])
             & batch.ap_mask[:, None, :])
    scan, cell, ap = owner.nonzero(as_tuple=True)
    own = batch.cell_aggregates.new_zeros(scans, aps, n_dwells, width)
    own[scan, ap, batch.cell_dwell[scan, cell]] = batch.cell_aggregates[scan, cell]

    # One step per AP per dwell: the channel, the AP's own cell, whether the
    # dwell was on its channel, and what share of the rivals sat there too.
    on_channel = tuned[:, None, :] == batch.ap_channel[:, :, None]
    share = ((on_channel & batch.ap_mask[:, :, None]).sum(dim=1)
             / batch.ap_mask.sum(dim=1, keepdim=True))
    steps = torch.cat([heard[:, None].expand(-1, aps, -1, -1), own,
                       on_channel[..., None].to(own.dtype),
                       share[:, None, :, None].expand(-1, aps, -1, -1).to(own.dtype)], dim=-1)
    return steps, dwell_exists
