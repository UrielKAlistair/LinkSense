"""The pieces the cell models share: the batch they all read, and the layers
the grid models are built from.

CellBatch is a batch of scans as train.py pads them, and every model takes one.
The layers follow the three stages of joint_ap_model.py:

  1. cell   FrameEncoder and CellEncoder, with CarrierSenseEncoder inside it,
            turn each cell's frames and aggregates, and a channel cell's
            carrier sense, into a token.
  2. grid   EncoderBlock, stacked, lets the cells attend to one another.
  3. APs    DecoderBlock lets question rows read the grid, through the mask
            cross_attention_bias builds.

joint_ap_model.py and target_ap_model.py build from all three stages, and
llm_model.py inherits them through JointAPModel. binned_model.py takes only the
batch and LOG_VAR_RANGE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.common.projection import DWELL_MS
from scripts.tf.cache_dataset import N_CATEGORIES

# Channels 36, 40, 44 and 48 sit four apart, one 20 MHz step each: a channel's
# index is its step from 36, and the distance between two falls in 0..3.
FIRST_CHANNEL = 36
CHANNEL_STEP = 4
N_CHANNELS = 4
MAX_CHANNEL_DISTANCE = N_CHANNELS - 1

# The predicted log variance is clamped to this range, which keeps exp(-log_var)
# in the loss finite.
LOG_VAR_RANGE = (-10.0, 10.0)

# Cells are encoded in this many groups of similar size, each padded only to
# its own largest cell.
CELL_GROUPS = 8

# How a token carries its tags. Under "add" the tags are summed into the finished
# cell vector, which is what the corpus was trained under. "quiet" and "loud"
# sum them into a normalised one and differ only in the scale the tables start
# at, which separates the normalising from the scale. The two "concat" modes
# give the tags TAG_WIDTH
# dimensions of their own and the content gives up that many; "concat_norm" also
# normalises the tag block, so neither half can shrink away from the other over
# training rather than only starting level with it.
# "add" is neither, "quiet" normalises only, "scale_only" rescales only, and
# "loud" does both: the four corners of normalising the content against raising
# the tag scale.
TAG_MODES = ("add", "quiet", "scale_only", "loud", "concat", "concat_norm", "codes")

# Under "codes" a tag is a fixed set of directions at a length the model tunes.
# The flat tags take mutually orthogonal codes, so their values are told apart
# exactly; the dwell is a time coordinate and takes a sinusoid, laid in the basis
# the flat codes leave over, so that nearby dwells stay near each other.
TAG_LENGTH = 4.0
TAG_WIDTH = 16

# Where in its dwell a frame began, as sine-cosine pairs whose periods fall
# geometrically from two dwells to 0.2 ms. Under "concat" they take the last
# POSITION_WIDTH columns of the frame token and the content keeps the rest; under
# "add" they span the whole width, at twice as many periods, and are summed onto
# a normalised content that has the whole width to itself.
FRAME_POSITIONS = ("concat", "add")
POSITION_WIDTH = 16
POSITION_PERIODS = (2.0, 0.2 / DWELL_MS)    # in dwells


def channel_index(channel: torch.Tensor) -> torch.Tensor:
    """Each channel's step from FIRST_CHANNEL. Padding carries channel 0 and
    reads as the first channel; it is masked wherever it could matter."""
    return ((channel - FIRST_CHANNEL) // CHANNEL_STEP).clamp(0, N_CHANNELS - 1)


@dataclass
class CellBatch:
    """One batch of scans, padded to the widest scan in it.

    G scans, C cells, M frames per cell, N frames in the batch, A discovered APs.

    frames             (N, F)     per-frame features, scan after scan, each
                                  scan's in the order heard
    frame_offsets      (N,)       where in its dwell each frame began, 0 to 1
    frame_categories   (N,)       management, control or data
    cell_members       (G, C, M)  rows of `frames`, padded with zero
    cell_member_mask   (G, C, M)  which of those indices are real
    cell_aggregates    (G, C, K)  over every frame of the cell, not only the M
    cell_cca           (G, C, S)  carrier sense, zero on AP cells
    cell_is_channel    (G, C)     a channel cell, else an AP cell
    cell_dwell         (G, C)     which of the 52 dwells
    cell_channel       (G, C)     the channel the radio was tuned to
    cell_ap_identity   (G, C)     identity of the AP an AP cell belongs to, else 0
    cell_mask          (G, C)     which cells are real
    descriptors        (G, A, D)  each discovered AP over the whole window
    ap_channel         (G, A)     the channel its beacons came from
    ap_identity        (G, A)     its identity, drawn afresh per sample in training
    ap_mask            (G, A)     which APs are real
    """

    frames: torch.Tensor
    frame_offsets: torch.Tensor
    frame_categories: torch.Tensor
    cell_members: torch.Tensor
    cell_member_mask: torch.Tensor
    cell_aggregates: torch.Tensor
    cell_cca: torch.Tensor
    cell_is_channel: torch.Tensor
    cell_dwell: torch.Tensor
    cell_channel: torch.Tensor
    cell_ap_identity: torch.Tensor
    cell_mask: torch.Tensor
    descriptors: torch.Tensor
    ap_channel: torch.Tensor
    ap_identity: torch.Tensor
    ap_mask: torch.Tensor


# ---------------------------------------------------------------------------
# 1. cell: one token per cell
# ---------------------------------------------------------------------------

# We embed each decoded frame, then group the frames by dwell, the 110 ms the
# radio spends on one channel. A dwell's frames form its channel cell, and each
# discovered AP on that channel gets a cell of the frames carrying its BSSID.
# The cell is the unit every model reads, and each becomes one 64-number token: its
# frame vectors attend to one another and are pooled, then joined with the
# cell's aggregates and, on a channel cell, its carrier sense.

class FrameEncoder(nn.Module):
    """One vector per decoded frame: what it was, and when in its dwell it began.

    What it was passes through an MLP over its features and category. When it
    began enters as a sinusoidal encoding, either set beside the content in its
    own columns or, having given the content the whole width and normalised it,
    summed onto it.
    """

    def __init__(self, n_features: int, width: int, position: str = "concat"):
        super().__init__()
        if position not in FRAME_POSITIONS:
            raise ValueError(f"position must be one of {FRAME_POSITIONS}, not {position!r}")
        self.position = position
        content = width if position == "add" else width - POSITION_WIDTH
        self.linear = nn.Linear(n_features, content)
        self.category = nn.Embedding(N_CATEGORIES, content)
        nn.init.normal_(self.category.weight, std=0.02)
        self.out = nn.Linear(content, content)
        # Summed, the content has to be held at a known size, or the sinusoid ends
        # up where the cell tags were: present and inaudible.
        self.norm = nn.LayerNorm(content) if position == "add" else None
        longest, shortest = POSITION_PERIODS
        # Summed, the sinusoid spans the whole width, so it samples the same range
        # of periods at twice as many of them.
        pairs = content // 2 if position == "add" else POSITION_WIDTH // 2
        periods = torch.logspace(math.log10(longest), math.log10(shortest), pairs)
        self.register_buffer("frequencies", 2 * math.pi / periods, persistent=False)

    def forward(self, frames: torch.Tensor, categories: torch.Tensor,
                offsets: torch.Tensor) -> torch.Tensor:
        content = self.out(F.gelu(self.linear(frames) + self.category(categories)))
        angles = offsets.unsqueeze(-1) * self.frequencies
        position = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if self.position == "add":
            return self.norm(content) + position
        return torch.cat([content, position], dim=-1)


class CodeBook(nn.Module):
    """One fixed vector per value of a tag, at a length the model tunes.

    The directions are set once and never trained. What a tag has to supply is
    that its values be told apart, and a fixed code supplies that exactly; the
    one thing left underdetermined is how loud the tag should be beside the
    content it is added to, and that is the parameter.
    """

    def __init__(self, codes: torch.Tensor, length: float = TAG_LENGTH):
        super().__init__()
        self.register_buffer("codes", codes / codes.norm(dim=-1, keepdim=True))
        self.length = nn.Parameter(torch.tensor(float(length)))

    def forward(self, index: torch.Tensor) -> torch.Tensor:
        return self.length * self.codes[index]


def orthogonal_codes(width: int, sizes: list[int], seed: int = 0
                     ) -> tuple[list[torch.Tensor], torch.Tensor]:
    """One orthonormal basis carved into a code set per size, and what is left.

    Carving from a single basis makes the tags orthogonal to each other as well
    as within themselves, so the sum of a cell's tags can be read apart again.
    """
    generator = torch.Generator().manual_seed(seed)
    basis, _ = torch.linalg.qr(torch.randn(width, width, generator=generator))
    sets, at = [], 0
    for size in sizes:
        sets.append(basis[at:at + size].clone())
        at += size
    return sets, basis[at:].clone()


def sinusoid_codes(n_values: int, basis: torch.Tensor) -> torch.Tensor:
    """A time coordinate as sine-cosine pairs, laid in the given basis.

    Periods fall geometrically from twice the range, so the slowest pair turns
    half a circle across it, to two steps, which is the fastest gap that can be
    told apart at all.
    """
    pairs = basis.shape[0] // 2
    periods = torch.logspace(math.log10(2 * n_values), math.log10(2.0), pairs)
    angles = torch.arange(n_values)[:, None] * (2 * math.pi / periods)
    return torch.cat([angles.sin(), angles.cos()], dim=-1) @ basis[:2 * pairs]


class CellEncoder(nn.Module):
    """One token per cell: its frames and aggregates, and a channel cell's
    carrier sense.

    The frames of a cell have no order that matters beyond the time each one
    carries, so they are read as a set: they attend to one another once, then
    collapse into a mean and an attention-weighted pick. The two kinds of cell
    share that attention, and each has its own pick and its own final layers.
    """

    def __init__(self, n_aggregates: int, width: int, heads: int, dropout: float,
                 out_width: int | None = None):
        super().__init__()
        # The frames arrive at `width`; out_width is what the cell token leaves
        # at, which is narrower when the tags are given dimensions of their own.
        out_width = width if out_width is None else out_width
        self.norm_attention = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout,
                                               batch_first=True)
        self.norm_pool = nn.LayerNorm(width)
        # One learned pool query per kind of cell, indexed by cell_is_channel.
        self.queries = nn.Parameter(torch.zeros(2, 1, width))
        nn.init.normal_(self.queries, std=0.02)
        self.pool_attention = nn.MultiheadAttention(width, heads, dropout=dropout,
                                                    batch_first=True)
        self.cca = CarrierSenseEncoder(width)
        self.empty = nn.Parameter(torch.zeros(width))
        nn.init.normal_(self.empty, std=0.02)

        def out(n_inputs: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(n_inputs), nn.Linear(n_inputs, width * 2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(width * 2, out_width))

        self.channel_out = out(3 * width + n_aggregates)
        self.ap_out = out(2 * width + n_aggregates)

    def forward(self, frames: torch.Tensor, batch: CellBatch) -> torch.Tensor:
        """(G, C, width) cell tokens from (N, width) frame vectors."""
        n_scans, n_cells, _ = batch.cell_members.shape

        # From here on the cells of every scan form one list.
        frame_rows = batch.cell_members.flatten(0, 1)
        is_member = batch.cell_member_mask.flatten(0, 1)
        is_channel = batch.cell_is_channel.flatten()
        aggregates = batch.cell_aggregates.flatten(0, 1)
        carrier_sense = batch.cell_cca.flatten(0, 1)
        frame_counts = is_member.sum(dim=1)

        # A cell holds anywhere from no frames to hundreds, so cells of similar
        # size are pooled together, each group padded only to its largest.
        pooled = frames.new_empty(n_scans * n_cells, 2 * frames.shape[-1])
        for group in frame_counts.argsort().chunk(CELL_GROUPS):
            longest = max(int(frame_counts[group].max()), 1)
            pooled[group] = self.pool(frames[frame_rows[group, :longest]],
                                      is_member[group, :longest], is_channel[group])

        # Channel cells join their carrier sense and aggregates, AP cells their
        # aggregates alone.
        is_ap = ~is_channel
        channel_tokens = self.channel_out(torch.cat(
            [pooled[is_channel], self.cca(carrier_sense[is_channel]),
             aggregates[is_channel]], dim=-1))
        ap_tokens = self.ap_out(torch.cat([pooled[is_ap], aggregates[is_ap]], dim=-1))
        tokens = channel_tokens.new_empty(n_scans * n_cells, channel_tokens.shape[-1])
        tokens[is_channel], tokens[is_ap] = channel_tokens, ap_tokens
        return tokens.reshape(n_scans, n_cells, -1)

    def pool(self, frames: torch.Tensor, is_member: torch.Tensor,
             is_channel: torch.Tensor) -> torch.Tensor:
        """A mean and an attention-weighted pick of each cell's frames, for a
        group of cells padded to the same number of frames."""
        is_padding = ~is_member
        # A cell that heard nothing has no frames to attend to. Letting it read
        # its first padding row keeps the attention defined; its pooled output
        # is replaced below, so what it reads there never survives.
        is_silent = is_padding.all(dim=1)
        is_padding[is_silent, 0] = False

        normed = self.norm_attention(frames)
        attended, _ = self.attention(normed, normed, normed,
                                     key_padding_mask=is_padding, need_weights=False)
        frames = self.norm_pool(frames + attended)

        frame_counts = is_member.sum(dim=1, keepdim=True).clamp(min=1)
        mean = (frames * is_member.unsqueeze(-1)).sum(dim=1) / frame_counts
        picked, _ = self.pool_attention(self.queries[is_channel.long()], frames, frames,
                                        key_padding_mask=is_padding, need_weights=False)
        pooled = torch.cat([mean, picked.squeeze(1)], dim=-1)
        # A silent cell's mean and pick are both the learned empty vector.
        return torch.where(is_silent[:, None], self.empty.repeat(2), pooled)


class CarrierSenseEncoder(nn.Module):
    """One vector per dwell from its 109 milliseconds of carrier sense.

    A convolution rather than a mean, because when the medium was busy within
    the dwell is what separates one long transmission from scattered short ones.
    """

    def __init__(self, width: int):
        super().__init__()
        self.conv = nn.Conv1d(1, 16, kernel_size=9, stride=4)
        self.out = nn.Linear(16 * 4, width)

    def forward(self, cca: torch.Tensor) -> torch.Tensor:
        shape = cca.shape[:-1]
        x = self.conv(cca.reshape(-1, 1, cca.shape[-1]))
        x = F.adaptive_avg_pool1d(F.gelu(x), 4).flatten(1)
        return self.out(x).reshape(*shape, -1)


# ---------------------------------------------------------------------------
# 2. grid: the cells attend to one another
# ---------------------------------------------------------------------------

# One block, stacked. Every cell sees every other cell of its scan, across both
# dwells and channels, so what a cell carries can depend on what the radio was
# hearing elsewhere in the window.

class EncoderBlock(nn.Module):
    """Pre-norm self-attention and feed-forward, the standard pair, with a
    learned number per head for each channel distance between two cells added
    to their attention score."""

    def __init__(self, width: int, heads: int, dropout: float,
                 distance_bias: bool = True):
        super().__init__()
        self.heads = heads
        self.norm_attention = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout,
                                               batch_first=True)
        self.norm_ff = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, width * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width * 4, width))
        self.distance_bias = (nn.Parameter(torch.zeros(heads, MAX_CHANNEL_DISTANCE + 1))
                              if distance_bias else None)

    def forward(self, tokens: torch.Tensor, cell_channel: torch.Tensor,
                cell_mask: torch.Tensor) -> torch.Tensor:
        if self.distance_bias is None:
            bias = padding_bias(cell_mask, self.heads, tokens.shape[1], tokens.dtype)
        else:
            bias = attention_mask(
                channel_distance_bias(self.distance_bias, cell_channel, cell_channel),
                cell_mask)
        normed = self.norm_attention(tokens)
        attended, _ = self.attention(normed, normed, normed,
                                     attn_mask=bias, need_weights=False)
        tokens = tokens + attended
        return tokens + self.ff(self.norm_ff(tokens))


class NativeGridReader(nn.ModuleList):
    """Stage 2 as a stack of EncoderBlocks, the grid reading itself.

    A ModuleList so that its blocks keep the names `encoder.0`, `encoder.1` that
    the models have always given them. llm_model.py offers the other reader; both
    take the same four arguments, so either can fill the stage.
    """

    def __init__(self, width: int, heads: int, dropout: float, layers: int,
                 distance_bias: bool = True):
        super().__init__([EncoderBlock(width, heads, dropout, distance_bias)
                          for _ in range(layers)])

    def forward(self, tokens: torch.Tensor, cell_dwell: torch.Tensor,
                cell_channel: torch.Tensor, cell_mask: torch.Tensor) -> torch.Tensor:
        """The dwell reaches a block only through the backbone reader, which
        positions cells by it; these blocks see the whole grid at once."""
        for block in self:
            tokens = block(tokens, cell_channel, cell_mask)
        return tokens


def channel_distance_bias(table: torch.Tensor, asking_channel: torch.Tensor,
                          read_channel: torch.Tensor) -> torch.Tensor:
    """Each head's learned number for how far each read token's channel lies
    from each asking token's, in 20 MHz steps: (B, heads, Q, C) from a
    (heads, MAX_CHANNEL_DISTANCE + 1) table, (B, Q) asking channels and (B, C)
    read ones."""
    distance = asking_channel[:, :, None] - read_channel[:, None, :]
    steps = (distance.abs() // CHANNEL_STEP).clamp(max=MAX_CHANNEL_DISTANCE)
    return table.t()[steps].permute(0, 3, 1, 2)


def attention_mask(bias: torch.Tensor, read_mask: torch.Tensor) -> torch.Tensor:
    """bias as MultiheadAttention's additive mask, (B * heads, Q, C), with the
    padded tokens barred. Padding travels in this float mask because attention
    takes a float mask and a boolean one as two different kinds, and mixing
    them is on its way out."""
    return bias.masked_fill((~read_mask)[:, None, None, :], float("-inf")).flatten(0, 1)


# ---------------------------------------------------------------------------
# 3. APs: one question row per discovered AP reads the grid
# ---------------------------------------------------------------------------

# Each discovered AP asks through a question row, which reads the grid by
# cross-attention. A learned bias shapes what a row reads: how far a cell's
# channel lies from the AP's, and whether the cell is the AP's own.

def cross_attention_bias(distance_bias: torch.Tensor, own_ap_bias: torch.Tensor,
                         ap_channel: torch.Tensor, ap_identity: torch.Tensor,
                         cell_channel: torch.Tensor, cell_is_channel: torch.Tensor,
                         cell_ap_identity: torch.Tensor,
                         cell_mask: torch.Tensor) -> torch.Tensor:
    """Per head, how far each cell's channel is from each asking AP's, whether
    the cell is that AP's own, and which cells are padding.

    ap_* are (B, Q) for Q asking APs, and cell_* are (B, C) for the grid they
    read.
    """
    bias = channel_distance_bias(distance_bias, ap_channel, cell_channel)
    own = ((~cell_is_channel)[:, None, :]
           & (cell_ap_identity[:, None, :] == ap_identity[:, :, None]))
    return attention_mask(bias + own[:, None] * own_ap_bias[:, None, None], cell_mask)


def padding_bias(cell_mask: torch.Tensor, heads: int, n_queries: int,
                 dtype: torch.dtype) -> torch.Tensor:
    """cross_attention_bias with both learned terms removed: padding is barred and
    nothing else is said. The ablation that asks whether the tags alone carry what
    the biases hand the model."""
    shape = (cell_mask.shape[0], heads, n_queries, cell_mask.shape[1])
    return attention_mask(torch.zeros(shape, dtype=dtype, device=cell_mask.device),
                          cell_mask)


class DecoderBlock(nn.Module):
    """Question rows compare notes, then read the grid.

    The rows attend to each other first, so an AP's row can read its rivals'
    rows: their descriptors, and after the first block what they read from the
    grid. Rivals also reach a row through the grid, whose cells have read the
    whole scan. With `compare` false a block skips that step, and its rows read
    the grid alone.
    """

    def __init__(self, width: int, heads: int, dropout: float, compare: bool = True):
        super().__init__()
        self.compare = compare
        if compare:
            self.norm_self = nn.LayerNorm(width)
            self.self_attention = nn.MultiheadAttention(width, heads, dropout=dropout,
                                                        batch_first=True)
        self.norm_query = nn.LayerNorm(width)
        self.norm_grid = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, heads, dropout=dropout,
                                                     batch_first=True)
        self.norm_ff = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, width * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width * 4, width))

    def forward(self, rows: torch.Tensor, grid: torch.Tensor,
                row_padding: torch.Tensor | None, bias: torch.Tensor) -> torch.Tensor:
        if self.compare:
            normed = self.norm_self(rows)
            talked, _ = self.self_attention(normed, normed, normed,
                                            key_padding_mask=row_padding, need_weights=False)
            rows = rows + talked

        # bias also bars the grid's padding.
        grid = self.norm_grid(grid)
        read, _ = self.cross_attention(self.norm_query(rows), grid, grid,
                                       attn_mask=bias, need_weights=False)
        rows = rows + read
        return rows + self.ff(self.norm_ff(rows))
