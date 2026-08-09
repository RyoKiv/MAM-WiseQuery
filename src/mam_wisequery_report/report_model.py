"""Minimal inference-only MAM-WiseQuery model used by the public release."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import LlamaForCausalLM, LlamaTokenizer

from .dynamic_prompt import MultiTokenAttributePromptGeneratorModified
from .mam import MultiLevelAggregationModule
from .position import OrderedImagePositionEmbedding
from .vendor.qformer import BertConfig, BertLMHeadModel
from .vision import RETFoundReportEncoder

LOGGER = logging.getLogger(__name__)


# 保留原工程 LayerNorm 的 float32 计算/原 dtype 返回语义，避免混合精度时与训练前向产生偏差。
class StableLayerNorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_dtype = inputs.dtype
        normalized = super().forward(inputs.to(torch.float32))
        return normalized.to(original_dtype)


# 新增报告专用模型，仅构建 RETFound→MAM→顺序位置编码→split-query Q-Former→动态 Prompt→LLaMA 前向，完全不初始化分类/对齐/训练损失模块。
class ReportOnlyMAMWiseQuery(nn.Module):
    """Checkpoint-compatible model containing only the report-generation graph."""

    def __init__(
        self,
        *,
        llama_model_path: str | Path,
        qformer_config_dir: str | Path,
        image_size: int = 224,
        max_txt_len: int = 150,
        max_context_len: int = 3800,
        prompt_template: str = "[INST] {} [/INST]",
        num_classes: int = 17,
        num_query_tokens: int = 32,
        num_prompts: int = 4,
    ):
        super().__init__()
        self.max_txt_len = int(max_txt_len)
        self.max_context_len = int(max_context_len)
        self.prompt_template = prompt_template
        self.num_classes = int(num_classes)
        self.num_disease_queries = int(num_classes)
        self.num_generic_queries = int(num_query_tokens - num_classes)
        self.llm_hidden_dim = 4096
        self.qformer_input_dim = 1408
        self.qformer_output_dim = 768
        self.mam_select_layer = -2

        self._init_llama(llama_model_path)
        self._init_visual_path(image_size)
        self._init_qformer(qformer_config_dir, num_query_tokens)
        self._init_split_queries(num_query_tokens)
        self._init_report_projection(num_prompts)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # 按原训练参数构建 Chinese-LLaMA-2-7B 和 q/v LoRA，语言模型基座仍由用户根据原许可证单独提供。
    def _init_llama(self, llama_model_path: str | Path) -> None:
        path = str(Path(llama_model_path).expanduser().resolve())
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(path, use_fast=False)
        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        llama_model = LlamaForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
        )
        llama_model = prepare_model_for_kbit_training(llama_model)
        llama_lora = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        )
        self.llama_model = get_peft_model(llama_model, llama_lora)
        self.llm_hidden_dim = int(self.llama_model.config.hidden_size)

    # 构建不读取外部视觉权重的 RETFound/MAM/顺序位置编码结构，后续由单一发布 checkpoint 一次性填充。
    def _init_visual_path(self, image_size: int) -> None:
        self.visual_encoder = RETFoundReportEncoder(image_size=image_size)
        self.ln_vision = StableLayerNorm(self.qformer_input_dim)
        self.vit_to_qformer_proj = nn.Linear(1024, self.qformer_input_dim)
        self.position_embedding = OrderedImagePositionEmbedding(feature_dim=1024)
        self.mam = MultiLevelAggregationModule(
            hidden_dim=1024,
            num_layers=24,
            num_heads=8,
            internal_ablation="full",
            select_layer_only=False,
            use_layer_embedding=True,
            use_image_global_gate=True,
            use_branch_gate=True,
            disable_branches=[],
        )

    # 使用随发布代码附带的小型 BERT config 构建 Q-Former，再按 Stage-1 相同配置包装 LoRA。
    def _init_qformer(
        self, qformer_config_dir: str | Path, num_query_tokens: int
    ) -> None:
        config = BertConfig.from_pretrained(str(qformer_config_dir))
        config.encoder_width = self.qformer_input_dim
        config.add_cross_attention = True
        config.cross_attention_freq = 2
        config.query_length = int(num_query_tokens)

        qformer = BertLMHeadModel(config=config)
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=config.initializer_range)

        qformer_lora = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=[
                "attention.self.query",
                "attention.self.value",
                "crossattention.self.query",
                "crossattention.self.key",
                "crossattention.self.value",
                "crossattention.output.dense",
            ],
        )
        self.Qformer = get_peft_model(qformer, qformer_lora)

    # 按训练 checkpoint 的精确键名与形状构建 17 个疾病 query 和其余 generic query，初值只用于结构占位并会被发布权重覆盖。
    def _init_split_queries(self, num_query_tokens: int) -> None:
        hidden_size = self.qformer_output_dim
        base_query_bank = self.query_tokens.detach().clone().squeeze(0)
        disease_indices = torch.arange(self.num_disease_queries, dtype=torch.long)
        generic_indices = torch.arange(
            self.num_disease_queries, num_query_tokens, dtype=torch.long
        )

        self.register_buffer("disease_embeddings", torch.zeros(self.num_classes, 768))
        self.register_buffer("blip_query_bank", base_query_bank)
        self.register_buffer("disease_base_query_indices", disease_indices)
        self.register_buffer("generic_base_query_indices", generic_indices)
        self.register_buffer(
            "disease_base_queries", base_query_bank[disease_indices].clone()
        )
        self.register_buffer(
            "selected_normal_query_index", torch.tensor(0, dtype=torch.long)
        )
        self.register_buffer("disease_sem_scale", torch.tensor(1.0))
        self.register_buffer("disease_sem_strength", torch.tensor(0.5))

        self.disease_sem_proj = nn.Linear(768, hidden_size)
        self.disease_query_residual = nn.Parameter(
            torch.zeros(1, self.num_disease_queries, hidden_size)
        )
        self.generic_query_tokens = nn.Parameter(
            base_query_bank[generic_indices].unsqueeze(0).clone()
        )
        self.query_type_embedding = nn.Embedding(2, hidden_size)
        self.disease_id_embedding = nn.Embedding(
            self.num_disease_queries, hidden_size
        )

    # 仅构建报告分支的 Q-Former→LLaMA 投影、疾病文本投影与最终 modified QueryNorm 动态 Prompt generator。
    def _init_report_projection(self, num_prompts: int) -> None:
        self.llama_proj = nn.Linear(self.qformer_output_dim, self.llm_hidden_dim)
        self.disease_to_llm_proj = nn.Linear(768, self.llm_hidden_dim)
        self.attribute_text_embeddings = None
        self.register_buffer("_attr_text_embeds", None, persistent=False)
        self.prompt_generator = MultiTokenAttributePromptGeneratorModified(
            visual_dim=self.qformer_output_dim,
            text_dim=self.llm_hidden_dim,
            num_attributes=self.num_classes,
            num_prompts=num_prompts,
            hidden_dim=512,
            num_heads=8,
            dropout=0.1,
            use_gamma=True,
            use_beta=True,
        )

    def maybe_autocast(self, dtype=torch.bfloat16):
        if self.device.type == "cuda":
            return torch.cuda.amp.autocast(dtype=dtype)
        return contextlib.nullcontext()

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.llama_model.base_model.model.model.embed_tokens(token_ids)

    # 保留原始 MiniGPT prompt/image 交织算法，但只处理推理时的单轮报告指令。
    def prompt_wrap(
        self,
        image_embeds: torch.Tensor,
        image_attention: torch.Tensor,
        prompts: str | Sequence[str] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prompts is None or len(prompts) == 0:
            return image_embeds, image_attention
        if isinstance(prompts, str):
            prompts = [prompts] * len(image_embeds)

        wrapped_batches = []
        for each_image_embed, each_prompt in zip(image_embeds, prompts):
            patch_count = each_image_embed.shape[-2]
            segments = each_prompt.split("<ImageHere>")
            if len(segments) <= 1:
                raise ValueError(
                    "The report prompt must contain the '<ImageHere>' placeholder"
                )

            interleaved = []
            for index, segment in enumerate(segments[:-1]):
                tokens = self.llama_tokenizer(
                    segment, return_tensors="pt", add_special_tokens=False
                ).to(image_embeds.device)
                text_embed = self.embed_tokens(tokens.input_ids.long())
                image_slice = each_image_embed[None][
                    :, index * patch_count : (index + 1) * patch_count
                ]
                interleaved.append(torch.cat([text_embed, image_slice], dim=1))
            wrapped = torch.cat(interleaved, dim=1)
            tail_tokens = self.llama_tokenizer(
                segments[-1], return_tensors="pt", add_special_tokens=False
            ).to(image_embeds.device)
            wrapped = torch.cat(
                [wrapped, self.embed_tokens(tail_tokens.input_ids.long())], dim=1
            )
            wrapped_batches.append(wrapped)

        lengths = [embedding.shape[1] for embedding in wrapped_batches]
        pad_embed = self.embed_tokens(
            torch.tensor(
                self.llama_tokenizer.pad_token_id, device=image_embeds.device
            )
        )
        max_length = min(max(lengths), self.max_context_len)
        output = pad_embed.expand(len(lengths), max_length, -1).clone()
        attention = torch.zeros(
            len(lengths), max_length, dtype=torch.int, device=image_embeds.device
        )
        for index, embedding in enumerate(wrapped_batches):
            length = min(lengths[index], self.max_context_len)
            output[index, :length] = embedding[:, :length]
            attention[index, :length] = 1
        return output, attention

    # 实现与训练模型完全同形的多层视觉特征提取，两张图像按列表位置 0/1 添加可学习嵌入后再展平。
    def extract_visual_features(
        self, images: torch.Tensor, position_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if images.dim() != 5 or images.shape[1] != 2:
            raise ValueError(
                f"Expected image tensor [B,2,C,H,W], got {tuple(images.shape)}"
            )
        batch_size, view_count, channels, height, width = images.shape
        if position_ids is None:
            position_ids = torch.tensor(
                [[0, 1]] * batch_size, dtype=torch.long, device=images.device
            )
        image_flat = images.reshape(
            batch_size * view_count, channels, height, width
        )

        with self.maybe_autocast():
            with torch.no_grad():
                _, hidden_states = (
                    self.visual_encoder.model.forward_features_with_hidden_states(
                        image_flat.to(self.visual_encoder.device)
                    )
                )
            patch_hidden_states = [state[:, 1:].detach() for state in hidden_states]
            image_features = self.mam(patch_hidden_states, self.mam_select_layer)

        image_features = image_features.reshape(
            batch_size, view_count, -1, image_features.shape[-1]
        )
        image_features = self.position_embedding(image_features, position_ids)
        image_features = image_features.flatten(1, 2)
        return self.ln_vision(self.vit_to_qformer_proj(image_features))

    def _build_split_query_tokens(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        hidden_size = self.disease_base_queries.size(-1)
        disease_text = self.disease_embeddings.to(device=device, dtype=torch.float32)
        disease_semantic = self.disease_sem_proj(disease_text)
        disease_semantic = F.layer_norm(disease_semantic, (hidden_size,))
        disease_semantic = (
            self.disease_sem_strength.float()
            * self.disease_sem_scale.float()
            * disease_semantic
        ).to(dtype=dtype)

        disease_type = self.query_type_embedding(
            torch.zeros(self.num_disease_queries, dtype=torch.long, device=device)
        ).to(dtype)
        disease_id = self.disease_id_embedding(
            torch.arange(
                self.num_disease_queries, dtype=torch.long, device=device
            )
        ).to(dtype)
        disease_queries = F.layer_norm(
            self.disease_base_queries.to(device=device, dtype=dtype)
            + self.disease_query_residual.to(device=device, dtype=dtype).squeeze(0)
            + disease_semantic
            + disease_type
            + disease_id,
            (hidden_size,),
        ).unsqueeze(0).expand(batch_size, -1, -1)

        generic_type = self.query_type_embedding(
            torch.ones(self.num_generic_queries, dtype=torch.long, device=device)
        ).to(dtype)
        generic_queries = F.layer_norm(
            self.generic_query_tokens.to(device=device, dtype=dtype)
            + generic_type.unsqueeze(0),
            (hidden_size,),
        ).expand(batch_size, -1, -1)
        return torch.cat([disease_queries, generic_queries], dim=1)

    # Q-Former 前向只返回动态 Prompt 和 LLaMA 投影所需特征，不构建也不计算分类 logits。
    def encode_img(
        self, images: torch.Tensor, position_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_embeds = self.extract_visual_features(images, position_ids)
        if not torch.isfinite(image_embeds).all():
            image_embeds = torch.nan_to_num(
                image_embeds, nan=0.0, posinf=1e4, neginf=-1e4
            )

        qformer_dtype = next(self.Qformer.parameters()).dtype
        image_embeds = image_embeds.to(dtype=qformer_dtype)
        image_attention = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
        )
        query_tokens = self._build_split_query_tokens(
            image_embeds.shape[0], image_embeds.device, qformer_dtype
        )
        with torch.cuda.amp.autocast(enabled=False):
            qformer_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention,
                return_dict=True,
                output_attentions=True,
            ).last_hidden_state
        if not torch.isfinite(qformer_output).all():
            qformer_output = torch.nan_to_num(
                qformer_output, nan=0.0, posinf=1e4, neginf=-1e4
            )

        llm_embeds = self.llama_proj(qformer_output)
        llm_attention = torch.ones(
            llm_embeds.size()[:-1], dtype=torch.long, device=llm_embeds.device
        )
        return llm_embeds, llm_attention, qformer_output

    def get_attribute_text_embeddings(self) -> torch.Tensor:
        return self.disease_to_llm_proj(self.disease_embeddings)

    def _dynamic_prompt(
        self, qformer_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attribute_embeddings = self.get_attribute_text_embeddings().to(
            device=qformer_output.device, dtype=qformer_output.dtype
        )
        with self.maybe_autocast():
            prompt_embeds = self.prompt_generator(
                qformer_output, attribute_embeddings
            )
        prompt_attention = torch.ones(
            prompt_embeds.size()[:-1],
            dtype=torch.long,
            device=prompt_embeds.device,
        )
        return prompt_embeds, prompt_attention

    def _normalize_prompt(self, prompt: str) -> str:
        prompt = str(prompt).strip()
        if not prompt:
            raise ValueError("Report prompt cannot be empty")
        instruction_start = prompt.rfind("[INST]")
        if instruction_start != -1:
            instruction_end = prompt.find("[/INST]", instruction_start)
            if instruction_end > instruction_start:
                prompt = prompt[instruction_start + len("[INST]") : instruction_end].strip()
        if prompt.startswith("<s>"):
            prompt = prompt[3:].strip()
        if "<ImageHere>" not in prompt:
            prompt = f"<Img><ImageHere></Img> {prompt}".strip()
        return self.prompt_template.format(prompt)

    def _decode(self, outputs: torch.Tensor) -> list[str]:
        reports = []
        for output_tokens in outputs:
            if output_tokens[0] == 0:
                output_tokens = output_tokens[1:]
            text = self.llama_tokenizer.decode(
                output_tokens, skip_special_tokens=True
            )
            text = text.split("</s>")[0].replace("<s>", "")
            reports.append(text.split(r"[/INST]")[-1].strip())
        return reports

    # 实现最终报告生成，固定将有序两图经 WiseQuery 编码后与动态 Prompt 和文本指令拼接，公开返回值仅为报告字符串列表。
    @torch.inference_mode()
    def generate(
        self,
        images: torch.Tensor,
        texts: Sequence[str],
        position_ids: torch.Tensor | None = None,
        num_beams: int = 1,
        max_new_tokens: int = 100,
        min_length: int = 1,
        top_p: float = 0.9,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> list[str]:
        if position_ids is None:
            position_ids = torch.tensor(
                [[0, 1]] * images.shape[0], dtype=torch.long, device=images.device
            )
        llm_embeds, llm_attention, qformer_output = self.encode_img(
            images, position_ids=position_ids
        )
        dynamic_embeds, dynamic_attention = self._dynamic_prompt(qformer_output)
        prompts = [self._normalize_prompt(prompt) for prompt in texts]
        condition_embeds, condition_attention = self.prompt_wrap(
            llm_embeds, llm_attention, prompts
        )

        batch_size = condition_embeds.shape[0]
        bos = torch.full(
            (batch_size, 1),
            self.llama_tokenizer.bos_token_id,
            dtype=torch.long,
            device=images.device,
        )
        bos_embeds = self.embed_tokens(bos)
        bos_attention = torch.ones(
            batch_size, 1, dtype=torch.long, device=images.device
        )
        inputs_embeds = torch.cat(
            [bos_embeds, dynamic_embeds, condition_embeds], dim=1
        )
        attention_mask = torch.cat(
            [bos_attention, dynamic_attention, condition_attention], dim=1
        )

        with self.maybe_autocast():
            outputs = self.llama_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                min_length=min_length,
                top_p=top_p,
                temperature=temperature,
                do_sample=do_sample,
            )
        return self._decode(outputs)

