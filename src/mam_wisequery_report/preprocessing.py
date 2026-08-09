"""Ordered two-image input resolution and OCT preprocessing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from .settings import PreprocessSettings


# 新增与原评估链一致的确定性图像处理器，不使用训练增强或随机裁剪。
class OCTImagePreprocessor:
    def __init__(self, settings: PreprocessSettings):
        self.settings = settings
        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambda image: image.convert("RGB")),
                transforms.Resize(
                    (settings.image_size, settings.image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(settings.mean, settings.std),
            ]
        )

    def __call__(self, image_path: str | Path) -> torch.Tensor:
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Input image does not exist: {path}")
        with Image.open(path) as image:
            return self.transform(image)


# 严格要求两张输入图像，并按 CLI 或数据记录中的原列表顺序堆叠为 [1,2,C,H,W]。
def load_ordered_image_pair(
    image_paths: Sequence[str | Path],
    preprocessor: OCTImagePreprocessor,
) -> torch.Tensor:
    paths = tuple(Path(path).expanduser().resolve() for path in image_paths)
    if len(paths) != 2:
        raise ValueError(f"Exactly two input images are required, got {len(paths)}")
    tensors = [preprocessor(path) for path in paths]
    return torch.stack(tensors, dim=0).unsqueeze(0)


# 新增按本地 JSON 记录取图的输入方式，只读取 folder/img 定位图像且原样保留 img 列表顺序。
def resolve_pair_from_dataset_record(
    dataset_json: str | Path,
    *,
    image_root: str | Path,
    split: str,
    record_index: int,
) -> tuple[Path, Path]:
    dataset_path = Path(dataset_json).expanduser().resolve()
    with dataset_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        if split not in payload:
            raise KeyError(f"Split {split!r} is absent from {dataset_path}")
        records = payload[split]
    elif isinstance(payload, list):
        records = payload
    else:
        raise TypeError("Dataset JSON must contain a record list or split-to-list mapping")

    if not isinstance(records, list):
        raise TypeError(f"Dataset split {split!r} must be a list")
    try:
        record = records[record_index]
    except IndexError as exc:
        raise IndexError(
            f"record_index={record_index} is outside split {split!r} with {len(records)} records"
        ) from exc

    if not isinstance(record, dict):
        raise TypeError(f"Record {record_index} must be an object")
    image_names = record.get("img")
    if not isinstance(image_names, list) or len(image_names) != 2:
        raise ValueError(
            f"Record {record_index} must contain exactly two entries in 'img'"
        )

    root = Path(image_root).expanduser().resolve()
    folder = str(record.get("folder", "")).strip()
    resolved: list[Path] = []
    for image_name in image_names:
        image_path = Path(str(image_name))
        if not image_path.is_absolute():
            image_path = root / folder / image_path
        resolved.append(image_path.resolve())

    missing = [path for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Dataset record points to missing image files:\n  - "
            + "\n  - ".join(str(path) for path in missing)
        )
    return resolved[0], resolved[1]

