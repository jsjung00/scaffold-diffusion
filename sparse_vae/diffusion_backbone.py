from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys 
from pathlib import Path 
sys.path.append(str(Path(__file__).parent))

from modules.utils import convert_module_to_f16, convert_module_to_f32
from modules.transformer import AbsolutePositionEmbedder, ModulatedTransformerBlock
from modules.spatial import patchify, unpatchify

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: a 1-D Tensor of N indices, one per batch element.
                These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class SparseStructureDiffusionModel(nn.Module):
    def __init__(
        self,
        backbone_config, 
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        self.backbone_config = backbone_config 
        #TODO: need to figure out and change the config resolution
        self.resolution = backbone_config.resolution
        self.in_channels = backbone_config.in_channels
        self.model_channels = backbone_config.model_channels
        self.cond_channels = backbone_config.cond_channels
        self.out_channels = backbone_config.out_channels
        self.num_blocks = backbone_config.num_blocks
        self.num_heads = backbone_config.num_heads 
        self.mlp_ratio = backbone_config.mlp_ratio
        self.patch_size = backbone_config.patch_size
        self.pe_mode = backbone_config.pe_mode
        self.use_fp16 = backbone_config.use_fp16
        self.qk_rms_norm = backbone_config.qk_rms_norm
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if self.use_fp16 else torch.float32

        self.t_embedder = TimestepEmbedder(self.model_channels)
        if self.share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.model_channels, 6 * self.model_channels, bias=True)
            )

        if self.pe_mode == "ape":
            pos_embedder = AbsolutePositionEmbedder(self.model_channels, 3)
            coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [self.resolution // self.patch_size] * 3], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            pos_emb = pos_embedder(coords)
            self.register_buffer("pos_emb", pos_emb)

        self.input_layer = nn.Linear(self.in_channels * self.patch_size**3, self.model_channels)

        self.blocks = nn.ModuleList([
            ModulatedTransformerBlock(
                self.model_channels, 
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode="full",
                use_checkpoint = self.use_checkpoint,
                use_rope = (self.pe_mode == "rope"),
                qk_rms_norm = self.qk_rms_norm,
                share_mod = self.share_mod, 
            )
            for _ in range(self.num_blocks)
        ])

        self.out_layer = nn.Linear(self.model_channels, self.out_channels * self.patch_size**3)

        self.initialize_weights()
        if self.use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
                f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"

        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()

        h = self.input_layer(h)
        h = h + self.pos_emb[None]
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        h = h.type(self.dtype)

        for block in self.blocks:
            h = block(h, t_emb)
        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)

        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
        h = unpatchify(h, self.patch_size).contiguous()

        return h, None 