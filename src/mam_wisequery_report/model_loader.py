"""Construction of the consolidated report-generation model."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from .checkpoint import is_report_state_key, load_release_payload_from_parts
from .settings import ReleaseSettings

LOGGER = logging.getLogger(__name__)


# 新增模型加载审计结果，让 CLI 能显式报告实际模式、权重和兼容键统计。
@dataclass(frozen=True)
class ModelLoadAudit:
    mode: str
    parts_manifest: str
    missing_key_count: int
    unexpected_key_count: int
    metadata: Mapping[str, Any]


# 统一设定 Python、NumPy、PyTorch 和 CUDA 随机种子，使默认非采样生成路径可重复。
def set_inference_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 仅从二分片清单加载发布状态，验证报告前向键后允许外部语言模型基座参数缺失。
def _load_release_parts(settings: ReleaseSettings):
    manifest_path = settings.model.release_parts_manifest
    if manifest_path is None:
        raise ValueError("release_parts_manifest is required for inference")
    payload = load_release_payload_from_parts(manifest_path)
    release_state = payload["model"]

    # 发布模式改用本包内的报告专用结构，不再 import 原工程的训练模型、数据集注册器或分类模块。
    from .report_model import ReportOnlyMAMWiseQuery

    model = ReportOnlyMAMWiseQuery(
        llama_model_path=settings.model.llama_model,
        qformer_config_dir=settings.model.qformer_config_dir,
        image_size=settings.preprocess.image_size,
        max_txt_len=150,
        prompt_template="[INST] {} [/INST]",
        num_classes=17,
        num_query_tokens=32,
        num_prompts=4,
    )
    required_model_keys = {
        key for key in model.state_dict() if is_report_state_key(key)
    }
    missing_required = sorted(required_model_keys.difference(release_state))
    if missing_required:
        raise RuntimeError(
            "Release checkpoint is missing tensors required by the report forward path: "
            + ", ".join(missing_required[:12])
        )

    message = model.load_state_dict(release_state, strict=False)
    unexpected = list(message.unexpected_keys)
    if unexpected:
        raise RuntimeError(
            "Release checkpoint contains keys not accepted by the model: "
            + ", ".join(unexpected[:12])
        )
    audit = ModelLoadAudit(
        mode="release",
        parts_manifest=str(manifest_path),
        missing_key_count=len(message.missing_keys),
        unexpected_key_count=0,
        metadata=payload.get("metadata", {}),
    )
    return model, audit


# 统一模型加载入口固定构建发布模型，完成权重校验后设为 eval 并迁移到目标设备。
def load_report_model(settings: ReleaseSettings):
    settings.validate(check_files=True)
    set_inference_seed(settings.runtime.seed)
    device = torch.device(settings.runtime.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {device} was requested, but CUDA is not available"
        )

    model, audit = _load_release_parts(settings)
    model.eval()
    model.to(device)
    LOGGER.info(
        "Loaded report model mode=%s parts_manifest=%s missing=%d unexpected=%d",
        audit.mode,
        audit.parts_manifest,
        audit.missing_key_count,
        audit.unexpected_key_count,
    )
    return model, audit
