"""What every model over the cell grid shares, and the two hooks that separate them.

A scan reaches these models as a grid of cells: one per dwell for the channel the
radio was tuned to, and one per dwell for each discovered AP on that channel.
Nothing else could have been observed, so nothing else is a token.

PROCESS, the same three stages in every subclass

  1. cell   `encode_cells` turns each cell's frames, its aggregates and, on a
            channel cell, its carrier sense into one token, then tags that token
            with what kind of cell it is, which dwell it covers and which channel
            it was heard on.
  2. grid   `self.encoder` lets the cells attend to one another. This is the one
            stage that swaps: NativeGridReader, or the backbone reader in
            llm_model.py. Both take the same arguments, so either fills it.
  3. APs    question rows read the grid through cross-attention, and the head
            turns each row into a mean and a log variance.

Subclasses differ in two places and nowhere else:

  `identity_tables`  the tag that says which AP a cell belongs to. The joint
                     model gives every cell its AP's identity, which does not
                     depend on the question being asked. The target model stamps
                     each cell's role relative to one AP, which does.
  `forward`          how many copies of the grid stage 2 runs over, and how many
                     questions ride on each copy. A tag that depends on the
                     question forces one copy per question.

Everything else - the cell encoder, the grid reader, the decoder blocks, the
cross-attention bias and the head - is built here once and inherited.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from scripts.tf.layers import (LOG_VAR_RANGE, MAX_CHANNEL_DISTANCE, N_CHANNELS,
                               TAG_LENGTH, TAG_MODES, CellBatch,
                               CellEncoder, CodeBook, DecoderBlock, FrameEncoder,
                               NativeGridReader, channel_index,
                               cross_attention_bias, orthogonal_tags, padding_bias,
                               sinusoid_tags)


class CellGridModel(nn.Module):
    """A transformer over the cells of a scan; a subclass decides who asks."""

    # Whether the question rows attend to one another before reading the grid.
    # A model carrying one question per grid has no rivals to confer with.
    compare = True

    def __init__(self, n_frame_features: int, n_aggregates: int, n_descriptor: int,
                 n_dwells: int = 52, width: int = 64, heads: int = 4,
                 cell_heads: int = 2, layers: int = 2, decoder_layers: int = 2,
                 dropout: float = 0.1, readout_bias: bool = True,
                 encoder_bias: bool = True, tag_mode: str = "codes",
                 tag_scale: float | None = None, grid_reader=None):
        super().__init__()
        if tag_mode not in TAG_MODES:
            raise ValueError(f"tag_mode must be one of {TAG_MODES}, not {tag_mode!r}")
        if tag_mode == "learned" and tag_scale is not None:
            raise ValueError("tag_scale applies only to fixed tags")
        self.heads = heads
        self.readout_bias = readout_bias
        self.frames = FrameEncoder(n_frame_features, width)
        self.cells = CellEncoder(n_aggregates, width, cell_heads, dropout,
                                 out_width=width)

        if tag_mode == "codes":
            # Fixed directions, one learnable length each. The flat tags take
            # mutually orthogonal directions and the dwell the basis they leave over.
            (flat_is_channel, flat_channel, flat_identity), spare = orthogonal_tags(
                width, [2, N_CHANNELS, self.identity_size()])
            length = tag_scale if tag_scale is not None else TAG_LENGTH / width ** 0.5
            self.is_channel = CodeBook(flat_is_channel, length)
            self.channel = CodeBook(flat_channel, length)
            self.dwell = CodeBook(sinusoid_tags(n_dwells, spare), length)
            self.make_identity_tags(flat_identity, length)
        else:
            self.is_channel = nn.Embedding(2, width)
            self.dwell = nn.Embedding(n_dwells, width)
            self.channel = nn.Embedding(N_CHANNELS, width)
            # The subclass's identity tag is made here, so that every table is
            # drawn from one point in the seeded stream whichever model is built.
            tables = [self.is_channel, self.dwell, self.channel,
                      *self.make_identity_tables(width)]
            for table in tables:
                nn.init.normal_(table.weight, std=0.02)

        self.encoder = (NativeGridReader(width, heads, dropout, layers, encoder_bias)
                        if grid_reader is None else grid_reader(width))
        self.decoder = nn.ModuleList(
            [DecoderBlock(width, heads, dropout, self.compare)
             for _ in range(decoder_layers)])

        self.question = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.normal_(self.question, std=0.02)
        self.descriptor = nn.Sequential(
            nn.Linear(n_descriptor, width), nn.GELU(),
            nn.Linear(width, width))
        # Learned scalars added to the cross-attention logits, per head: one per
        # channel distance, so a question can weigh its own channel against its
        # neighbours', and one for the cells of the question's own AP.
        if readout_bias:
            self.distance_bias = nn.Parameter(
                torch.zeros(heads, MAX_CHANNEL_DISTANCE + 1))
            self.own_ap_bias = nn.Parameter(torch.zeros(heads))

        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, 2))

    # -----------------------------------------------------------------------
    # The two hooks
    # -----------------------------------------------------------------------

    def make_identity_tables(self, tag_width: int) -> list[nn.Embedding]:
        """The subclass's own tag tables, assigned to self and returned so this
        class can initialise them alongside the three it made itself."""
        raise NotImplementedError

    def identity_size(self) -> int:
        """How many values the subclass's own tag takes, so this class can
        reserve that many tags for it."""
        raise NotImplementedError

    def make_identity_tags(self, tags: torch.Tensor, length: float) -> None:
        """Assign the subclass's own tag, from the tags reserved for it."""
        raise NotImplementedError

    def cell_identity(self, batch: CellBatch) -> torch.Tensor | None:
        """Whose cell this is, for a model that can say so without knowing which
        AP is being asked about. None from a model whose answer depends on the
        question, which then tags its own copies of the grid."""
        return None

    def forward(self, batch: CellBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and log variance of the standardised log1p throughput, one pair
        per discovered AP, laid out (scans, APs)."""
        raise NotImplementedError

    # -----------------------------------------------------------------------
    # The stages, shared
    # -----------------------------------------------------------------------

    def join(self, content: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        """Add identity and position tags to the content at its native scale."""
        return content + tags

    def cell_parts(self, batch: CellBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage 1 as far as every model agrees: each cell's content, and the tags
        for what kind it is, when it was heard and on which channel. They come
        back apart because a model tagging copies of the grid has one more tag to
        add before the two are joined."""
        content = self.cells(self.frames(batch.frames, batch.frame_categories,
                                         batch.frame_offsets), batch)
        tags = (self.is_channel(batch.cell_is_channel.long())
                + self.dwell(batch.cell_dwell)
                + self.channel(channel_index(batch.cell_channel)))
        return content, tags

    def encode_cells(self, batch: CellBatch) -> torch.Tensor:
        """Stage 1 finished, for a model whose identity tag stands independent of
        the question."""
        content, tags = self.cell_parts(batch)
        own = self.cell_identity(batch)
        return self.join(content, tags if own is None else tags + own)

    def read_grid(self, tokens: torch.Tensor, cell_dwell: torch.Tensor,
                  cell_channel: torch.Tensor, cell_mask: torch.Tensor) -> torch.Tensor:
        """Stage 2. Every argument is per grid copy, so a model running one copy
        per question passes its own expanded tensors and the reader is none the
        wiser."""
        return self.encoder(tokens, cell_dwell, cell_channel, cell_mask)

    def build_rows(self, descriptors: torch.Tensor, ap_channel: torch.Tensor,
                   tags: torch.Tensor | None) -> torch.Tensor:
        """The question rows: a learned seed, the AP's window descriptor, and the
        tags that put a row in the same dimensions as the cells it will read."""
        content = self.question + self.descriptor(descriptors)
        channel = self.channel(channel_index(ap_channel))
        return self.join(content, channel if tags is None else channel + tags)

    def readout_mask(self, ap_channel: torch.Tensor, ap_identity: torch.Tensor,
                     cell_channel: torch.Tensor, cell_is_channel: torch.Tensor,
                     cell_ap_identity: torch.Tensor, cell_mask: torch.Tensor,
                     dtype: torch.dtype) -> torch.Tensor:
        """What a question row may read, and what the learned biases say about it.
        With the biases ablated away, padding is barred and nothing else is said."""
        if not self.readout_bias:
            return padding_bias(cell_mask, self.heads, ap_channel.shape[1], dtype)
        return cross_attention_bias(self.distance_bias, self.own_ap_bias,
                                    ap_channel, ap_identity, cell_channel,
                                    cell_is_channel, cell_ap_identity, cell_mask)

    def score(self, rows: torch.Tensor, grid: torch.Tensor,
              row_padding: torch.Tensor | None, bias: torch.Tensor
              ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage 3: the rows read the grid, then the head reads the rows."""
        for block in self.decoder:
            rows = block(rows, grid, row_padding, bias)
        prediction = self.head(self.norm(rows))
        return prediction[..., 0], prediction[..., 1].clamp(*LOG_VAR_RANGE)
