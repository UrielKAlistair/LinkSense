#!/usr/bin/env python3
"""Stage 2 read by a pretrained language model instead of by encoder blocks.

INPUT   the grid of cell tokens, one copy per question the model is asking
OUTPUT  the same grid, the cells having attended to one another

This file holds a grid reader and nothing else. It takes the four arguments
NativeGridReader takes and returns what it returns, so either fills stage 2 of
any CellGridModel: the joint model and the target model both accept one through
`grid_reader`, and neither knows which it was handed.

Where JointAPModel runs two encoder blocks of width 64 over the cells, this hands
them to a frozen decoder-only language model, with low-rank adapters on its query
and value projections and one linear layer either side to cross between the
widths.

The backbone never sees text. Its token embedding table is replaced by a stub and
the cell tokens go in as input embeddings, so no tokenizer is involved and no
vocabulary is ever consulted.

Attention over the grid is block-causal. Within one dwell every cell attends to
every other cell of that dwell; across dwells attention only looks backwards. A
sweep evolves in time, so a cell having no access to its future is the honest
structure, and it leaves the model valid on a sweep that was cut short. The cells
of a single dwell are one channel cell and the APs heard on that channel, which
carry no order between them. Every cell of a dwell takes that dwell as its rotary
position, so the backbone cannot impose an order inside a dwell that the mask says
is symmetric.

Alpha, dropout and the choice of adapted projections come from NetLLM's released
code; the rank is ours.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModel

DEFAULT_BACKBONE = "meta-llama/Llama-3.2-1B"

# The backbone's frozen weights are held in bfloat16: at two bytes a parameter a
# 1B model is 2.5 GB, which leaves an 8 GB card room for activations, where
# single precision would not fit alongside training. The adapters and the two
# projections stay in single precision, which is what their optimiser steps need.
BACKBONE_DTYPE = torch.bfloat16

# Adapters go on the query and value projections alone. Rank is a command-line
# argument, since it is what sets the trainable budget.
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ("q_proj", "v_proj")


def llama_reader(backbone: str | None = None, lora_rank: int = 8,
                 pretrained: bool = True) -> Callable[[int], nn.Module]:
    """A grid reader waiting only to be told the model's width.

    CellGridModel builds its reader inside its own constructor, so that the
    seeded stream of initial weights runs in one order whichever reader is asked
    for. This hands it something it can call with the width and nothing else.
    """
    name = backbone or DEFAULT_BACKBONE
    return lambda width: LlamaGridReader(width, name, lora_rank, pretrained)


class LlamaGridReader(nn.Module):
    """The cells attending to one another inside a frozen language model."""

    def __init__(self, width: int, backbone: str, lora_rank: int, pretrained: bool):
        super().__init__()
        config = AutoConfig.from_pretrained(backbone)
        self.backbone = adapted_backbone(config, backbone, lora_rank, pretrained)
        self.into_backbone = nn.Linear(width, config.hidden_size)
        self.out_of_backbone = nn.Linear(config.hidden_size, width)

    def forward(self, tokens: torch.Tensor, cell_dwell: torch.Tensor,
                cell_channel: torch.Tensor, cell_mask: torch.Tensor) -> torch.Tensor:
        """The backbone reads the grid under a block-causal mask.

        Only the backbone's frozen weights are in BACKBONE_DTYPE, so the cast
        happens at its boundary and nowhere else: the projections either side run
        in the precision the cells arrive in. Training runs under autocast, which
        would cast anyway, but scoring on a CPU does not and would otherwise fail
        on the dtype. The channel reaches attention through the tags on the tokens
        rather than through a bias, since the backbone's blocks are not ours.
        """
        outside = tokens.dtype
        hidden = self.into_backbone(tokens).to(BACKBONE_DTYPE)
        mask = block_causal_mask(cell_dwell, cell_mask, BACKBONE_DTYPE)
        read = self.backbone(inputs_embeds=hidden, attention_mask=mask,
                             position_ids=cell_dwell).last_hidden_state
        return self.out_of_backbone(read.to(outside)).to(outside)


def adapted_backbone(config, name: str, lora_rank: int, pretrained: bool) -> nn.Module:
    """The frozen language model that reads the grid, with adapters attached.

    get_peft_model freezes everything it did not add, so the only weights left
    training in here are the adapters. With `pretrained` false the weights come
    from the config instead of the checkpoint: identical architecture, identical
    adapter budget, differing only in what the frozen matrices hold, which is the
    control for whether pretraining is what helped.
    """
    backbone = (AutoModel.from_pretrained(name, dtype=BACKBONE_DTYPE) if pretrained
                else AutoModel.from_config(config).to(BACKBONE_DTYPE))
    # Nothing here tokenises, so the embedding table is weight that would occupy
    # memory and never be read: inputs_embeds is the path the grid takes in.
    backbone.set_input_embeddings(nn.Identity())
    return get_peft_model(backbone, LoraConfig(
        r=lora_rank, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", target_modules=list(LORA_TARGETS)))


def block_causal_mask(cell_dwell: torch.Tensor, cell_mask: torch.Tensor,
                      dtype: torch.dtype) -> torch.Tensor:
    """An additive mask letting a cell attend to its own dwell and earlier ones.

    `cell_dwell` and `cell_mask` are both (B, C). Returns (B, 1, C, C), zero
    where attention is allowed and the dtype's lowest value where it is not,
    which is the four-dimensional form a decoder-only transformer takes as a mask
    to apply verbatim rather than as padding to fold into one it builds itself.

    A cell may always attend to itself, padding included. Without that, a padded
    cell's row can be masked in full; softmax over nothing is NaN, and that NaN
    reaches the AP rows even though their cross-attention weights padded cells at
    zero, because zero times NaN is NaN.
    """
    not_later = cell_dwell[:, None, :] <= cell_dwell[:, :, None]
    allowed = not_later & cell_mask[:, None, :]
    allowed |= torch.eye(allowed.shape[-1], dtype=torch.bool, device=allowed.device)
    mask = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
    return mask.masked_fill(~allowed, torch.finfo(dtype).min)[:, None]
