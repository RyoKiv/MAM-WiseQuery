"""Inference copy of the trained Multi-Level Aggregation Module."""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# 从训练模型固定提取 MAM 实现供报告专用运行时使用，参数名与已训练 checkpoint 保持一致。
class MultiLevelAggregationModule(nn.Module):
    """MAM: 多层视觉特征聚合模块。

    从 ViT 多层隐藏状态中稀疏选取 early / mid / late 三组代表层,
    先注入可学习的 Layer Embedding, 再用 Multi-Head Cross-Layer Attention 聚合。
    聚合后的各分支额外通过分支自身的 image-global summary 做 token-wise gating,
    然后进行分支归一化与可学习融合, 最后通过 select residual shortcut
    回到原始维度空间。

    灵感来源: vMLLM (Boosting Multi-modal LLM with Enhanced Visual Features)

    Args:
        hidden_dim: ViT 隐藏维度 (RETFound = 1024)
        num_layers: ViT 总层数 (RETFound = 24)
        num_heads: 层间聚合注意力头数
    """

    LAYER_GROUP_SPECS_1BASED = (
        (3, 6),
        (9, 12),
        (18, 24),
    )

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_layers: int = 24,
        num_heads: int = 8,
        internal_ablation: str = "full",
        select_layer_only: bool = False,
        use_layer_embedding: bool = True,
        use_image_global_gate: bool = True,
        use_branch_gate: bool = True,
        disable_branches=None,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim must be divisible by num_heads, got "
                f"hidden_dim={hidden_dim}, num_heads={num_heads}"
            )

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        # 接入 MAM 内部消融开关，默认保持 Full MAM 行为，仅在 yaml 显式设置时关闭对应分支或门控。
        self.internal_ablation = str(internal_ablation or "full")
        self.select_layer_only = bool(select_layer_only)
        self.use_layer_embedding = bool(use_layer_embedding)
        self.use_image_global_gate = bool(use_image_global_gate)
        self.use_branch_gate = bool(use_branch_gate)
        self.disable_branches = set(disable_branches or [])
        valid_branches = {"early", "mid", "late"}
        unknown_branches = self.disable_branches - valid_branches
        if unknown_branches:
            raise ValueError(
                f"Unknown MAM branches in disable_branches: {sorted(unknown_branches)}. "
                f"Valid branches are {sorted(valid_branches)}."
            )

        # 显式层身份编码: 让模块区分 "这是第几层"
        self.layer_embedding = nn.Embedding(num_layers, hidden_dim)

        # 三段式多头层间聚合
        self.early_q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.early_k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.early_v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.early_out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.mid_q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.mid_k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.mid_v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.mid_out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.late_q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.late_k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.late_v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.late_out_proj = nn.Linear(hidden_dim, hidden_dim)

        # vMLLM cls_attention 的图像全局门控版本:
        # 使用分支全局 summary 对自身 token 做 image-global gating
        self.global_q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.global_k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.global_v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.global_log_temp = nn.Parameter(torch.log(torch.tensor(4.0)))

        # 分支归一化 + 可学习融合门控
        self.branch_gate_logits = nn.Parameter(torch.full((4,), 2.0))

        # 融合投影: 4*D → D (early + mid + late + select residual)
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier 初始化所有投影层。"""
        nn.init.normal_(self.layer_embedding.weight, mean=0.0, std=0.02)
        for module in [
            self.early_q_proj, self.early_k_proj, self.early_v_proj,
            self.early_out_proj,
            self.mid_q_proj, self.mid_k_proj, self.mid_v_proj,
            self.mid_out_proj,
            self.late_q_proj, self.late_k_proj, self.late_v_proj,
            self.late_out_proj,
            self.global_q_proj, self.global_k_proj, self.global_v_proj,
        ]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        for module in self.fusion_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _normalize_branch(self, branch_feat: torch.Tensor) -> torch.Tensor:
        """对单个分支做 token-wise L2 normalize。"""
        normalized = F.normalize(branch_feat.float(), dim=-1, eps=1e-6)
        return normalized.to(branch_feat.dtype)

    def _add_layer_embeddings(
        self,
        hidden_states: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """为每一层 token 注入显式 layer identity。"""
        total_layers = len(hidden_states)
        if total_layers > self.num_layers:
            raise ValueError(
                f"hidden_states has {total_layers} layers, which exceeds "
                f"configured num_layers={self.num_layers} for layer embeddings."
            )

        layer_ids = torch.arange(total_layers, device=hidden_states[0].device)
        layer_embeds = self.layer_embedding(layer_ids)

        enriched_hidden_states = []
        for layer_idx, hidden_state in enumerate(hidden_states):
            layer_embed = layer_embeds[layer_idx].to(dtype=hidden_state.dtype)
            enriched_hidden_states.append(hidden_state + layer_embed.view(1, 1, -1))

        return enriched_hidden_states

    def _normalize_select_idx(self, select_idx: int, total_layers: int) -> int:
        """将负索引标准化为 [0, num_layers) 内的正索引。"""
        if select_idx < 0:
            select_idx += total_layers
        if not 0 <= select_idx < total_layers:
            raise ValueError(
                f"select_idx out of range: got {select_idx}, total_layers={total_layers}"
            )
        return select_idx

    def _split_layer_groups(self, hidden_states: List[torch.Tensor], select_idx: int):
        """按固定稀疏层位划分 early / mid / late，并移除 select 层。"""
        total_layers = len(hidden_states)
        max_required_layer = max(max(group) for group in self.LAYER_GROUP_SPECS_1BASED)
        if total_layers < max_required_layer:
            raise ValueError(
                "Sparse layer grouping requires enough hidden states: "
                f"need at least {max_required_layer}, got {total_layers}."
            )

        groups = []
        for group_spec in self.LAYER_GROUP_SPECS_1BASED:
            group_indices = [layer_id - 1 for layer_id in group_spec]
            group = [
                hidden_states[layer_idx]
                for layer_idx in group_indices
                if layer_idx != select_idx
            ]
            if not group:
                raise ValueError(
                    "select_layer removes an entire aggregation group. "
                    f"total_layers={total_layers}, select_idx={select_idx}, "
                    f"group_spec={group_spec}"
                )
            groups.append(group)
        return groups

    def _aggregate(
        self,
        hidden_states_list: List[torch.Tensor],
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        out_proj: nn.Linear,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """用多头层间注意力聚合多层隐藏状态。

        Args:
            hidden_states_list: L 个 [B, N, D] 张量
            q_proj, k_proj, v_proj, out_proj: 投影层

        Returns:
            聚合特征 [B, N, D]；return_attention=True 时额外返回 [B, N, L] 层间注意力。
        """
        # Stack: [B, N, L, D]
        stacked = torch.stack(hidden_states_list, dim=-2)
        # Query: 各层均值 [B, N, 1, D]
        query = stacked.mean(dim=-2, keepdim=True)

        batch_size, num_tokens, num_layers, _ = stacked.shape

        Q = q_proj(query).reshape(
            batch_size, num_tokens, 1, self.num_heads, self.head_dim
        ).permute(0, 1, 3, 2, 4)                   # [B, N, H, 1, Dh]
        K = k_proj(stacked).reshape(
            batch_size, num_tokens, num_layers, self.num_heads, self.head_dim
        ).permute(0, 1, 3, 2, 4)                   # [B, N, H, L, Dh]
        V = v_proj(stacked).reshape(
            batch_size, num_tokens, num_layers, self.num_heads, self.head_dim
        ).permute(0, 1, 3, 2, 4)                   # [B, N, H, L, Dh]

        with torch.autocast('cuda', enabled=False):
            Q_f32 = Q.float()
            K_f32 = K.float()
            # Scaled dot-product attention over the layer axis: [B, N, H, 1, L]
            scale = self.head_dim ** 0.5
            attn = torch.matmul(Q_f32, K_f32.transpose(-1, -2)) / scale
            attn = attn.nan_to_num()
            attn = attn.softmax(dim=-1)
            attn = attn.to(V.dtype)

        context = torch.matmul(attn, V)            # [B, N, H, 1, Dh]
        context = context.permute(0, 1, 3, 2, 4).reshape(
            batch_size, num_tokens, self.hidden_dim
        )
        context = out_proj(context)

        # 残差保留原始层均值，避免退化为单纯投影后的层间平均
        out = query.squeeze(-2) + context
        # 为可视化脚本可选返回 token-wise 层间 attention，默认 forward 行为保持只返回聚合特征。
        if return_attention:
            layer_attn = attn.squeeze(-2).mean(dim=2)
            return out, layer_attn
        return out

    def _aggregate_or_zero(
        self,
        branch_name: str,
        hidden_states_list: List[torch.Tensor],
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        out_proj: nn.Linear,
        reference_feat: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        # 为 early/mid/late 分支消融提供零占位，并在 diagnostics 模式同步返回空 attention 占位。
        if branch_name in self.disable_branches:
            if return_attention:
                return torch.zeros_like(reference_feat), None
            return torch.zeros_like(reference_feat)
        return self._aggregate(
            hidden_states_list,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            return_attention=return_attention,
        )

    def image_global_attention(
        self,
        seq_feat: torch.Tensor,
        global_feat: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor:
        """vMLLM cls_attention 的图像全局门控版本。"""
        if global_feat.dim() != 2:
            raise ValueError(
                f"global_feat must have shape [B, D], got {tuple(global_feat.shape)}"
            )

        q = F.normalize(
            self.global_q_proj(seq_feat).float(), dim=-1, eps=1e-6
        )                                                                  # [B, N, D]
        k = F.normalize(
            self.global_k_proj(global_feat).float(), dim=-1, eps=1e-6
        ).unsqueeze(-1)                                                    # [B, D, 1]
        v = self.global_v_proj(seq_feat)                                   # [B, N, D]

        with torch.autocast('cuda', enabled=False):
            global_temp = self.global_log_temp.float().exp().clamp(max=20.0)
            attn_scores = torch.matmul(q, k)                               # [B, N, 1]
            attn_scores = attn_scores.nan_to_num()
            attn_weights = torch.sigmoid(global_temp * attn_scores).to(v.dtype)

        weighted = attn_weights * v
        # 支持可选返回 image-global gate 的 token-wise 权重，用于 MAM 内部注意力图。
        if return_weights:
            return weighted, attn_weights.squeeze(-1)
        return weighted

    def _apply_branch_conditioning(
        self,
        seq_feat: torch.Tensor,
        global_feat: torch.Tensor = None,
        return_gate: bool = False,
    ) -> torch.Tensor:
        """对单个视觉分支施加图像全局门控。"""
        # 接入 w/o image-global gate 消融，并在 diagnostics 模式返回 gate 权重或空占位。
        if global_feat is None or not self.use_image_global_gate:
            if return_gate:
                return seq_feat, None
            return seq_feat
        if return_gate:
            gated_feat, gate_weights = self.image_global_attention(
                seq_feat,
                global_feat,
                return_weights=True,
            )
            return 0.5 * (seq_feat + gated_feat), gate_weights
        return 0.5 * (seq_feat + self.image_global_attention(seq_feat, global_feat))

    def forward(
        self,
        hidden_states: List[torch.Tensor],
        select_idx: int,
        return_diagnostics: bool = False,
    ) -> torch.Tensor:
        """多层聚合前向传播。

        Args:
            hidden_states: 24 个 [B, N, D] 张量, 每个 block 的输出
            select_idx: 选定残差层索引

        Returns:
            聚合后的视觉特征 [B, N, D]；return_diagnostics=True 时额外返回 MAM 内部注意力诊断。
        """
        raw_hidden_states = hidden_states
        select_idx = self._normalize_select_idx(select_idx, len(raw_hidden_states))
        select_shortcut = raw_hidden_states[select_idx]

        # 接入 select-layer-only 消融，直接使用原始选中层 shortcut，跳过多层聚合、门控和融合投影。
        if self.select_layer_only:
            # select-only 消融下仍提供最小 diagnostics 元数据，避免可视化脚本误判为运行失败。
            if return_diagnostics:
                diagnostics = {
                    "select_layer_only": True,
                    "select_idx": int(select_idx),
                    "select_layer": int(select_idx + 1),
                    "source_layers": {"early": [], "mid": [], "late": [], "select": [int(select_idx + 1)]},
                    "layer_attention": {},
                    "global_gate": {},
                    "branch_gates": None,
                    "disabled_branches": sorted(self.disable_branches),
                    "use_image_global_gate": bool(self.use_image_global_gate),
                    "use_branch_gate": bool(self.use_branch_gate),
                }
                return select_shortcut, diagnostics
            return select_shortcut

        # 接入 w/o layer embedding 消融，关闭时多层聚合直接使用原始 ViT hidden states。
        if self.use_layer_embedding:
            enriched_hidden_states = self._add_layer_embeddings(hidden_states)
        else:
            enriched_hidden_states = raw_hidden_states
        early_states, mid_states, late_states = self._split_layer_groups(
            enriched_hidden_states, select_idx
        )
        # 记录 MAM 三个稀疏层组的 1-based 源层编号，供诊断图标题和 metadata 使用。
        source_layers = {
            branch_name: [
                int(layer_id)
                for layer_id in group_spec
                if int(layer_id - 1) != int(select_idx)
            ]
            for branch_name, group_spec in zip(
                ("early", "mid", "late"),
                self.LAYER_GROUP_SPECS_1BASED,
            )
        }
        source_layers["select"] = [int(select_idx + 1)]
        # 聚合/门控使用带 layer identity 的特征；最终 residual 保留纯视觉 shortcut。
        select_layer_feat = enriched_hidden_states[select_idx]

        early_result = self._aggregate_or_zero(
            "early",
            early_states,
            self.early_q_proj,
            self.early_k_proj,
            self.early_v_proj,
            self.early_out_proj,
            select_shortcut,
            return_attention=return_diagnostics,
        )
        mid_result = self._aggregate_or_zero(
            "mid",
            mid_states,
            self.mid_q_proj,
            self.mid_k_proj,
            self.mid_v_proj,
            self.mid_out_proj,
            select_shortcut,
            return_attention=return_diagnostics,
        )
        late_result = self._aggregate_or_zero(
            "late",
            late_states,
            self.late_q_proj,
            self.late_k_proj,
            self.late_v_proj,
            self.late_out_proj,
            select_shortcut,
            return_attention=return_diagnostics,
        )
        # 在 diagnostics 模式拆出各分支层间 attention，普通模式保持原有张量路径。
        if return_diagnostics:
            early_feat, early_layer_attn = early_result
            mid_feat, mid_layer_attn = mid_result
            late_feat, late_layer_attn = late_result
        else:
            early_feat, mid_feat, late_feat = early_result, mid_result, late_result
            early_layer_attn = mid_layer_attn = late_layer_attn = None

        global_select = select_layer_feat.mean(dim=1)
        early_global_gate = mid_global_gate = late_global_gate = select_global_gate = None

        if "early" not in self.disable_branches:
            early_conditioned = self._apply_branch_conditioning(
                early_feat, early_feat.mean(dim=1), return_gate=return_diagnostics
            )
            if return_diagnostics:
                early_feat, early_global_gate = early_conditioned
            else:
                early_feat = early_conditioned
        if "mid" not in self.disable_branches:
            mid_conditioned = self._apply_branch_conditioning(
                mid_feat, mid_feat.mean(dim=1), return_gate=return_diagnostics
            )
            if return_diagnostics:
                mid_feat, mid_global_gate = mid_conditioned
            else:
                mid_feat = mid_conditioned
        if "late" not in self.disable_branches:
            late_conditioned = self._apply_branch_conditioning(
                late_feat, late_feat.mean(dim=1), return_gate=return_diagnostics
            )
            if return_diagnostics:
                late_feat, late_global_gate = late_conditioned
            else:
                late_feat = late_conditioned
        select_conditioned = self._apply_branch_conditioning(
            select_layer_feat, global_select, return_gate=return_diagnostics
        )
        if return_diagnostics:
            select_layer_feat, select_global_gate = select_conditioned
        else:
            select_layer_feat = select_conditioned

        early_feat = self._normalize_branch(early_feat)
        mid_feat = self._normalize_branch(mid_feat)
        late_feat = self._normalize_branch(late_feat)
        select_layer_feat = self._normalize_branch(select_layer_feat)

        # 接入 w/o branch gate 消融，关闭时使用等权分支，不读取可学习 branch_gate_logits。
        if self.use_branch_gate:
            branch_gates = torch.sigmoid(self.branch_gate_logits).to(dtype=early_feat.dtype)
        else:
            branch_gates = early_feat.new_ones(4)
        early_feat = early_feat * branch_gates[0]
        mid_feat = mid_feat * branch_gates[1]
        late_feat = late_feat * branch_gates[2]
        select_layer_feat = select_layer_feat * branch_gates[3]

        # Concat + Project: [B, N, 4D] → [B, N, D]
        fused = torch.cat(
            [early_feat, mid_feat, late_feat, select_layer_feat],
            dim=-1,
        )
        fusion_delta = self.fusion_proj(fused)
        output = select_shortcut + fusion_delta

        # 按需返回 MAM 内部可视化诊断，包含层间 attention、global gate 和四分支 gate。
        if return_diagnostics:
            diagnostics = {
                "select_layer_only": False,
                "select_idx": int(select_idx),
                "select_layer": int(select_idx + 1),
                "source_layers": source_layers,
                "layer_attention": {
                    "early": None if early_layer_attn is None else early_layer_attn.detach().float(),
                    "mid": None if mid_layer_attn is None else mid_layer_attn.detach().float(),
                    "late": None if late_layer_attn is None else late_layer_attn.detach().float(),
                },
                "global_gate": {
                    "early": None if early_global_gate is None else early_global_gate.detach().float(),
                    "mid": None if mid_global_gate is None else mid_global_gate.detach().float(),
                    "late": None if late_global_gate is None else late_global_gate.detach().float(),
                    "select": None if select_global_gate is None else select_global_gate.detach().float(),
                },
                "branch_gates": branch_gates.detach().float(),
                "disabled_branches": sorted(self.disable_branches),
                "use_image_global_gate": bool(self.use_image_global_gate),
                "use_branch_gate": bool(self.use_branch_gate),
            }
            return output, diagnostics
        return output



