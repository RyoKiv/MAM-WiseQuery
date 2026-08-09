"""Checkpoint-compatible RETFound encoder architecture without training weights."""

from __future__ import annotations

from functools import partial

import timm.models.vision_transformer
import torch
import torch.nn as nn


# 提取 RETFound 推理所需的 ViT-L/16 结构与 24 层隐状态输出，不再从独立 RETFound checkpoint 初始化。
class RETFoundVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, global_pool: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

    def forward_features_with_hidden_states(self, images: torch.Tensor):
        batch_size = images.shape[0]
        tokens = self.patch_embed(images)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed)

        hidden_states = []
        for block in self.blocks:
            tokens = block(tokens)
            hidden_states.append(tokens)
        return self.norm(tokens), hidden_states


# 以不带分类 head 的包装器保持 visual_encoder.model.* 键名，仅向 MAM 提供 patch 隐状态。
class RETFoundReportEncoder(nn.Module):
    def __init__(self, image_size: int = 224):
        super().__init__()
        self.model = RETFoundVisionTransformer(
            img_size=image_size,
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            num_classes=0,
            global_pool=False,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        self.num_features = 1024

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(self, images: torch.Tensor):
        _, hidden_states = self.model.forward_features_with_hidden_states(images)
        return hidden_states

