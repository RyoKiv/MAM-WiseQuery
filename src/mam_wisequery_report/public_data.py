"""Validated access to the anonymous public OCT demonstration manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# 新增公开样本对象，仅暴露匿名编号、有序双图路径和仅供检查的参考报告。
@dataclass(frozen=True)
class PublicSample:
    sample_id: str
    image_paths: tuple[Path, Path]
    reference_report: str


# 将 manifest 相对路径限制在其数据目录内，阻止路径穿越读取仓库外文件。
def _resolve_inside_data(data_root: Path, relative_path: str) -> Path:
    candidate = (data_root / relative_path).resolve()
    try:
        candidate.relative_to(data_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Public image path escapes the data directory: {relative_path}"
        ) from exc
    return candidate


# 公开样本仅按清单顺序读取两张现存图像，不再保存或校验图像 SHA-256；参考报告不进入模型输入。
def load_public_sample(
    sample_id: str,
    manifest_path: str | Path,
) -> PublicSample:
    manifest_path = Path(manifest_path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest: dict[str, Any] = json.load(handle)
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise TypeError("Public manifest must contain a 'samples' list")

    record = next((item for item in samples if item.get("id") == sample_id), None)
    if record is None:
        raise KeyError(f"Unknown public sample ID: {sample_id}")
    image_names = record.get("images")
    if not isinstance(image_names, list) or len(image_names) != 2:
        raise ValueError(f"Sample {sample_id} must contain exactly two ordered images")

    data_root = manifest_path.parent
    image_paths: list[Path] = []
    for relative_path in image_names:
        image_path = _resolve_inside_data(data_root, str(relative_path))
        if not image_path.is_file():
            raise FileNotFoundError(f"Public image does not exist: {image_path}")
        image_paths.append(image_path)

    report = str(record.get("report", "")).strip()
    if not report:
        raise ValueError(f"Sample {sample_id} has an empty reference report")
    return PublicSample(sample_id, (image_paths[0], image_paths[1]), report)
