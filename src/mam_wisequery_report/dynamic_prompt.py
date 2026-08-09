"""Report-time dynamic prompt generator used by the released checkpoint."""

import torch
import torch.nn as nn

# 仅保留最终 QueryNorm 报告路径使用的 RMSNorm 与 modified prompt generator，参数名和计算保持训练版一致。
class RMSNormWithScale(nn.Module):
    # 将可学习输出幅度改为形状为[1]的张量，保持 RMSNorm 幅度控制能力的同时，规避 AdamW foreach 对 0 维参数的广播报错。
    def __init__(self, dim, eps=1e-6, init_scale=1.0):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.output_scale = nn.Parameter(torch.ones(1) * float(init_scale))

    def forward(self, x):
        original_dtype = x.dtype
        x_float = x.float()
        rms = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(rms + self.eps)
        x_norm = x_norm * self.weight.float() * self.output_scale.float()
        return x_norm.to(dtype=original_dtype)


# 恢复 MultiTokenAttributePromptGenerator 为最初版实现，保留原始的 LayerNorm、纯 Cross-Attention 输出和逐通道 gamma/beta 调制逻辑，避免影响现有 import 与历史对照实验。

class MultiTokenAttributePromptGeneratorModified(nn.Module):
    def __init__(
        self,
        visual_dim=1024,
        text_dim=4096,
        num_attributes=16,
        num_prompts=4,
        hidden_dim=512,
        num_heads=8,
        dropout=0.1,
        use_gamma=True,
        use_beta=True,
    ):
        super().__init__()

        self.text_dim = text_dim
        self.num_attributes = num_attributes
        self.num_prompts = num_prompts
        # 为 modified 动态 Prompt generator 增加 gamma/beta 独立开关，支持在不改动主干结构的前提下做“缺少 gamma / 缺少 beta / 两者都缺少”的参数消融实验。
        self.use_gamma = bool(use_gamma)
        self.use_beta = bool(use_beta)

        self.base_codebook = nn.Parameter(
            torch.randn(num_attributes, num_prompts, text_dim)
        )
        nn.init.normal_(self.base_codebook, std=0.02)

        self.ln_vis_in = nn.LayerNorm(visual_dim)
        self.ln_query_in = nn.LayerNorm(text_dim)
        self.vis_proj = nn.Linear(visual_dim, text_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )

        self.ln_attn_out = nn.LayerNorm(text_dim)

        # 将 gamma/beta 调制从逐通道 [B, A, P, D] 还原为 prompt-wise 标量 [B, A, P, 1]，让每条 prompt 重新共享单一缩放与平移系数。
        self.gamma_mlp = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_prompts),
        )

        self.beta_mlp = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_prompts),
        )

        # 保持 gamma/beta 输出层零初始化，使单通道 prompt-wise 调制在训练初期接近 identity transform，避免直接破坏 base_codebook。
        nn.init.zeros_(self.gamma_mlp[-1].weight)
        nn.init.zeros_(self.gamma_mlp[-1].bias)
        nn.init.zeros_(self.beta_mlp[-1].weight)
        nn.init.zeros_(self.beta_mlp[-1].bias)

        # 将最终输出归一化从 LayerNorm 改为带可学习幅度的 RMSNorm，更贴近 LLaMA 系 embedding 的归一化风格并允许保留可控幅度。
        self.final_norm = RMSNormWithScale(text_dim)

    def forward(self, visual_features, attribute_text_embeddings):
        B = visual_features.size(0)
        device = visual_features.device
        dtype = visual_features.dtype

        visual_features = self.ln_vis_in(visual_features)
        visual_features_proj = self.vis_proj(visual_features)

        if attribute_text_embeddings.dim() == 2:
            text_queries = attribute_text_embeddings.unsqueeze(0).expand(B, -1, -1)
        else:
            text_queries = attribute_text_embeddings
        text_queries = text_queries.to(dtype=dtype, device=device)

        # 在增强版中保留 query 侧归一化，先对 attribute query 做分布对齐，再送入跨模态注意力以稳定视觉条件更新。
        text_queries = self.ln_query_in(text_queries)

        # Cross-Attention 后保留 attribute query 残差并执行 Add & Norm，避免视觉结果直接覆盖属性锚点语义。
        attn_out, _ = self.cross_attn(
            query=text_queries,
            key=visual_features_proj,
            value=visual_features_proj,
        )

        attended_visuals = self.ln_attn_out(text_queries + attn_out)

        gamma_flat = self.gamma_mlp(attended_visuals)
        beta_flat = self.beta_mlp(attended_visuals)

        # 将 gamma/beta reshape 回单通道 [B, A, P, 1]，通过广播对整条 prompt 向量施加统一缩放和平移。
        gamma = gamma_flat.view(B, self.num_attributes, self.num_prompts, 1)
        beta = beta_flat.view(B, self.num_attributes, self.num_prompts, 1)

        # 缩小 gamma/beta 的初期调制幅度，让模型先在 base prompt 附近稳定训练，再逐步学习更强的自适应偏移。
        # 根据消融开关决定是否启用 gamma/beta 调制；关闭时分别退化为 identity scaling 和 zero shift，从而精确比较各分量缺失对性能的影响。
        if self.use_gamma:
            gamma = 1.0 + 0.1 * torch.tanh(gamma)
        else:
            gamma = torch.ones_like(gamma)
        if self.use_beta:
            beta = 0.1 * torch.tanh(beta)
        else:
            beta = torch.zeros_like(beta)

        base_prompts = self.base_codebook.unsqueeze(0).expand(B, -1, -1, -1)
        base_prompts = base_prompts.to(dtype=dtype, device=device)

        dynamic_prompts = base_prompts * gamma + beta

        prompt_sequence = dynamic_prompts.view(B, -1, self.text_dim)

        # 使用 RMSNormWithScale 对输出 prompt 做最终归一化与幅度校准，使其更贴近下游 LLaMA 系 embedding 分布。
        prompt_sequence = self.final_norm(prompt_sequence)

        return prompt_sequence


# 基于低秩 delta prompt 的动态 Prompt 生成器。

