"""Learned order-position embedding for the two-image input list."""

import torch
import torch.nn as nn


# 保留训练时的三位置参数形状，公开接口只按输入列表第一/第二位置使用前两个索引。
class OrderedImagePositionEmbedding(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.scan_embed = nn.Parameter(torch.zeros(3, 1, feature_dim))
        nn.init.trunc_normal_(self.scan_embed, std=0.02)

    def forward(
        self, image_features: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        return image_features + self.scan_embed[position_ids]

