"""Typed configuration for the public inference package."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml


# 模型路径仅保留二分片清单、外部语言模型基座和随代码提供的 Q-Former 配置目录。
@dataclass(frozen=True)
class ModelPaths:
    release_parts_manifest: Optional[Path]
    llama_model: Path
    qformer_config_dir: Path


# 新增确定性 OCT 预处理参数，与原始评估处理器的 RGB、Resize 和归一化设置保持一致。
@dataclass(frozen=True)
class PreprocessSettings:
    image_size: int = 224
    mean: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    std: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)


# 新增报告生成参数配置，默认使用可重复的贪心解码且不包含分类输出字段。
@dataclass(frozen=True)
class GenerationSettings:
    prompt: str
    num_beams: int = 1
    max_new_tokens: int = 120
    min_length: int = 1
    top_p: float = 0.9
    temperature: float = 1.0
    do_sample: bool = False


# 运行时仅保留设备与随机种子，公开推理固定加载合并后的发布权重。
@dataclass(frozen=True)
class RuntimeSettings:
    device: str = "cuda:0"
    seed: int = 42


# 配置加载、CLI 覆盖与运行前校验统一面向二分片清单，不再接受完整 checkpoint 路径。
@dataclass(frozen=True)
class ReleaseSettings:
    release_root: Path
    model: ModelPaths
    preprocess: PreprocessSettings
    generation: GenerationSettings
    runtime: RuntimeSettings

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "ReleaseSettings":
        config_path = Path(config_path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        if not isinstance(raw, Mapping):
            raise TypeError(f"Configuration root must be a mapping: {config_path}")

        release_root = config_path.parent.parent
        model_raw = _mapping(raw, "model")
        preprocess_raw = _mapping(raw, "preprocess")
        generation_raw = _mapping(raw, "generation")
        runtime_raw = _mapping(raw, "runtime")

        model = ModelPaths(
            release_parts_manifest=_optional_path(
                model_raw.get("release_parts_manifest"), release_root
            ),
            llama_model=_path(_required(model_raw, "llama_model"), release_root),
            qformer_config_dir=_path(
                _required(model_raw, "qformer_config_dir"), release_root
            ),
        )
        preprocess = PreprocessSettings(
            image_size=int(preprocess_raw.get("image_size", 224)),
            mean=_float_triplet(preprocess_raw.get("mean"), PreprocessSettings.mean),
            std=_float_triplet(preprocess_raw.get("std"), PreprocessSettings.std),
        )
        generation = GenerationSettings(
            prompt=str(_required(generation_raw, "prompt")),
            num_beams=int(generation_raw.get("num_beams", 1)),
            max_new_tokens=int(generation_raw.get("max_new_tokens", 120)),
            min_length=int(generation_raw.get("min_length", 1)),
            top_p=float(generation_raw.get("top_p", 0.9)),
            temperature=float(generation_raw.get("temperature", 1.0)),
            do_sample=bool(generation_raw.get("do_sample", False)),
        )
        runtime = RuntimeSettings(
            device=str(runtime_raw.get("device", "cuda:0")),
            seed=int(runtime_raw.get("seed", 42)),
        )
        settings = cls(
            release_root=release_root,
            model=model,
            preprocess=preprocess,
            generation=generation,
            runtime=runtime,
        )
        settings.validate(check_files=False)
        return settings

    def with_overrides(
        self,
        *,
        parts_manifest: Optional[str | Path] = None,
        llama_model: Optional[str | Path] = None,
        device: Optional[str] = None,
        prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        num_beams: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> "ReleaseSettings":
        model = self.model
        if parts_manifest is not None:
            manifest_path = _path(parts_manifest, self.release_root)
            model = replace(model, release_parts_manifest=manifest_path)
        if llama_model is not None:
            model = replace(model, llama_model=_path(llama_model, self.release_root))

        generation = replace(
            self.generation,
            prompt=self.generation.prompt if prompt is None else str(prompt),
            max_new_tokens=(
                self.generation.max_new_tokens
                if max_new_tokens is None
                else int(max_new_tokens)
            ),
            num_beams=self.generation.num_beams
            if num_beams is None
            else int(num_beams),
            top_p=self.generation.top_p if top_p is None else float(top_p),
            temperature=(
                self.generation.temperature
                if temperature is None
                else float(temperature)
            ),
            do_sample=self.generation.do_sample
            if do_sample is None
            else bool(do_sample),
        )
        runtime = replace(
            self.runtime,
            device=self.runtime.device if device is None else str(device),
        )
        updated = replace(self, model=model, generation=generation, runtime=runtime)
        updated.validate(check_files=False)
        return updated

    def validate(self, *, check_files: bool = True) -> None:
        if self.preprocess.image_size <= 0:
            raise ValueError("preprocess.image_size must be positive")
        if self.generation.max_new_tokens <= 0:
            raise ValueError("generation.max_new_tokens must be positive")
        if self.generation.num_beams <= 0:
            raise ValueError("generation.num_beams must be positive")
        if not self.generation.prompt.strip():
            raise ValueError("generation.prompt cannot be empty")
        if not check_files:
            return

        # 运行前只校验二分片清单、外部 LLaMA 基座与 Q-Former 配置目录。
        required = {
            "llama_model": self.model.llama_model,
            "qformer_config_dir": self.model.qformer_config_dir,
            "release_parts_manifest": self.model.release_parts_manifest,
        }

        missing = []
        for name, path in required.items():
            if path is None or _contains_unexpanded_variable(path) or not path.exists():
                missing.append(f"{name}={path}")
        if missing:
            raise FileNotFoundError(
                "Required inference paths are missing or unresolved:\n  - "
                + "\n  - ".join(missing)
            )


# 将 YAML 节点严格限定为 mapping，避免错误配置在模型初始化后才暴露。
def _mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"Configuration section {key!r} must be a mapping")
    return value


def _required(raw: Mapping[str, Any], key: str) -> Any:
    value = raw.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise KeyError(f"Missing required configuration value: {key}")
    return value


# 统一展开用户目录和环境变量，并以明确基准目录解析所有相对路径。
def _path(value: str | Path, base: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=False)


def _optional_path(value: Any, base: Path) -> Optional[Path]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _path(value, base)


def _contains_unexpanded_variable(path: Path) -> bool:
    return "$" in str(path)


# 对归一化统计量执行三通道校验，防止静默广播导致预处理偏差。
def _float_triplet(
    value: Any, default: tuple[float, float, float]
) -> tuple[float, float, float]:
    if value is None:
        return default
    values = tuple(float(item) for item in value)
    if len(values) != 3:
        raise ValueError(f"Expected three channel values, got {values}")
    return values
