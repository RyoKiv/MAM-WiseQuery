"""Validation and direct two-part loading for the release checkpoint."""

from __future__ import annotations

import hashlib
import io
import json
from bisect import bisect_right
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

FORMAT_NAME = "mam-wisequery-report-inference"
FORMAT_VERSION = 1
PARTS_FORMAT_NAME = "mam-wisequery-report-release-parts"
PARTS_FORMAT_VERSION = 1

# 定义报告生成前向必需的状态前缀，排除分类头、对齐头、训练损失和优化器状态。
REPORT_STATE_PREFIXES = (
    "visual_encoder.",
    "ln_vision.",
    "vit_to_qformer_proj.",
    "position_embedding.",
    "mam.",
    "Qformer.",
    "query_tokens",
    "disease_query_residual",
    "generic_query_tokens",
    "disease_embeddings",
    "blip_query_bank",
    "disease_base_query_indices",
    "generic_base_query_indices",
    "disease_base_queries",
    "selected_normal_query_index",
    "disease_sem_scale",
    "disease_sem_strength",
    "disease_sem_proj.",
    "query_type_embedding.",
    "disease_id_embedding.",
    "llama_proj.",
    "disease_to_llm_proj.",
    "prompt_generator.",
)

REQUIRED_GROUPS = {
    "vision": (
        "visual_encoder.",
        "ln_vision.",
        "vit_to_qformer_proj.",
        "position_embedding.",
        "mam.",
    ),
    "qformer": ("Qformer.", "query_tokens"),
    "split_queries": (
        "disease_query_residual",
        "generic_query_tokens",
        "disease_embeddings",
        "blip_query_bank",
        "disease_base_query_indices",
        "generic_base_query_indices",
        "disease_base_queries",
        "selected_normal_query_index",
        "disease_sem_scale",
        "disease_sem_strength",
        "disease_sem_proj.",
        "query_type_embedding.",
        "disease_id_embedding.",
    ),
    "llama_projection": ("llama_proj.",),
    "dynamic_prompt": ("prompt_generator.", "disease_to_llm_proj."),
    "llama_lora": ("llama_model.",),
}


# 限定 LLaMA 仅发布 LoRA 适配器，避免重复打包外部语言模型基座。
def is_report_state_key(key: str) -> bool:
    # 发布权重不保留 RETFound 原分类 head，报告运行时只构建不带 head 的视觉编码器。
    if key.startswith("visual_encoder.model.head."):
        return False
    if key.startswith(REPORT_STATE_PREFIXES):
        return True
    return key.startswith("llama_model.") and ".lora_" in key


# 对发布权重进行结构完整性检查，防止将简单键名合并但缺少 Q-Former 基座的文件误当成可独立推理权重。
def validate_report_state_dict(state_dict: Mapping[str, torch.Tensor]) -> None:
    if not state_dict:
        raise ValueError("Release state_dict is empty")

    unexpected = [key for key in state_dict if not is_report_state_key(key)]
    if unexpected:
        raise ValueError(
            "Release state_dict contains non-inference keys: "
            + ", ".join(unexpected[:8])
        )

    missing_groups = []
    keys = tuple(state_dict)
    for group, prefixes in REQUIRED_GROUPS.items():
        if group == "llama_lora":
            present = any(
                key.startswith("llama_model.") and ".lora_" in key for key in keys
            )
        else:
            present = all(
                any(key.startswith(prefix) for key in keys) for prefix in prefixes
            )
        if not present:
            missing_groups.append(group)
    if missing_groups:
        raise ValueError(
            "Release state_dict is missing required inference groups: "
            + ", ".join(missing_groups)
        )

    qformer_base_keys = [
        key for key in keys if key.startswith("Qformer.") and ".lora_" not in key
    ]
    if len(qformer_base_keys) < 100:
        raise ValueError(
            "Release checkpoint does not contain a complete Q-Former base; "
            f"found only {len(qformer_base_keys)} non-LoRA Q-Former tensors"
        )


# 将有序二分片映射为可 seek 的虚拟连续文件，使 torch.load 无需在磁盘生成完整 checkpoint。
class _ConcatenatedCheckpointReader(io.BufferedIOBase):
    def __init__(self, part_paths: Sequence[Path]):
        self._files = [path.open("rb") for path in part_paths]
        self._sizes = [path.stat().st_size for path in part_paths]
        self._starts = [0]
        for size in self._sizes:
            self._starts.append(self._starts[-1] + size)
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._starts[-1] + offset
        else:
            raise ValueError(f"Unsupported seek mode: {whence}")
        if position < 0:
            raise ValueError("Cannot seek to a negative checkpoint position")
        self._position = min(position, self._starts[-1])
        return self._position

    def read(self, size: int = -1) -> bytes:
        remaining = self._starts[-1] - self._position
        requested = remaining if size is None or size < 0 else min(size, remaining)
        chunks: list[bytes] = []
        while requested > 0:
            part_index = bisect_right(self._starts, self._position) - 1
            if part_index >= len(self._files):
                break
            local_offset = self._position - self._starts[part_index]
            read_size = min(requested, self._sizes[part_index] - local_offset)
            self._files[part_index].seek(local_offset)
            chunk = self._files[part_index].read(read_size)
            if not chunk:
                break
            chunks.append(chunk)
            self._position += len(chunk)
            requested -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        if not self.closed:
            for handle in self._files:
                handle.close()
        super().close()


# 严格校验分片清单、路径边界、文件大小与 SHA-256，并返回清单规定顺序的两个权重分片。
def _load_verified_part_paths(manifest_path: Path) -> list[Path]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, Mapping):
        raise TypeError(f"Checkpoint parts manifest must be a mapping: {manifest_path}")
    if manifest.get("format") != PARTS_FORMAT_NAME:
        raise ValueError(f"Unsupported checkpoint parts manifest: {manifest_path}")
    if int(manifest.get("version", -1)) != PARTS_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint parts version: {manifest.get('version')}"
        )

    parts = manifest.get("parts")
    if (
        manifest.get("part_count") != 2
        or not isinstance(parts, list)
        or len(parts) != 2
    ):
        raise ValueError("Checkpoint manifest must describe exactly two ordered parts")

    root = manifest_path.parent.resolve()
    part_paths: list[Path] = []
    total_size = 0
    for index, part in enumerate(parts):
        if not isinstance(part, Mapping):
            raise TypeError(f"Checkpoint part {index} metadata must be a mapping")
        filename = part.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            raise TypeError(f"Checkpoint part {index} filename must be non-empty")
        part_path = (root / filename).resolve()
        try:
            part_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Checkpoint part escapes weights directory: {filename}"
            ) from exc
        if not part_path.is_file():
            raise FileNotFoundError(f"Checkpoint part is missing: {part_path}")

        expected_size = part.get("size_bytes")
        if (
            not isinstance(expected_size, int)
            or part_path.stat().st_size != expected_size
        ):
            raise RuntimeError(f"Checkpoint part {index} has an unexpected file size")
        expected_sha256 = part.get("sha256")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError(f"Checkpoint part {index} has an invalid SHA-256 value")
        try:
            int(expected_sha256, 16)
        except ValueError as exc:
            raise ValueError(
                f"Checkpoint part {index} has a non-hexadecimal SHA-256 value"
            ) from exc
        if sha256_file(part_path) != expected_sha256.lower():
            raise RuntimeError(f"Checkpoint part {index} failed SHA-256 verification")
        part_paths.append(part_path)
        total_size += expected_size

    assembled = manifest.get("assembled")
    if not isinstance(assembled, Mapping) or assembled.get("size_bytes") != total_size:
        raise ValueError("Checkpoint manifest has inconsistent assembled size metadata")
    return part_paths


# 删除完整 checkpoint 兼容入口，发布模型只允许通过二分片清单校验并跨分片安全反序列化。
def load_release_payload_from_parts(manifest_path: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    part_paths = _load_verified_part_paths(manifest_path)
    with _ConcatenatedCheckpointReader(part_paths) as reader:
        payload = torch.load(reader, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint payload must be a mapping: {manifest_path}")
    if payload.get("format") != FORMAT_NAME:
        raise ValueError(
            f"Not a {FORMAT_NAME!r} release checkpoint: {manifest_path}. "
            "Download both GitHub Release parts into weights/."
        )
    if int(payload.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported release checkpoint version: {payload.get('format_version')}"
        )
    state_dict = payload.get("model")
    if not isinstance(state_dict, Mapping):
        raise TypeError("Release checkpoint is missing the 'model' state_dict")
    validate_report_state_dict(state_dict)
    return dict(payload)


# 新增分块 SHA-256 计算，用于权重下载校验与论文附件的可追溯记录。
def sha256_file(path: str | Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
