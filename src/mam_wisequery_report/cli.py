"""Command-line interface for report-only inference."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from .pipeline import ReportGenerationPipeline
from .preprocessing import resolve_pair_from_dataset_record
from .public_data import load_public_sample
from .settings import ReleaseSettings


# 公开 CLI 仅允许二分片清单覆盖，并继续按记录顺序支持三种双图输入方式。
def build_parser() -> argparse.ArgumentParser:
    release_root = Path(__file__).resolve().parents[2]
    default_config = release_root / "configs" / "inference.yaml"
    default_manifest = release_root / "data" / "samples.json"
    parser = argparse.ArgumentParser(
        description="Generate one OCT report from an ordered pair of OCT images."
    )
    parser.add_argument("--config", type=Path, default=default_config)

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--images",
        nargs=2,
        metavar=("IMAGE_0", "IMAGE_1"),
        help="Two image paths in the same order as the local data record.",
    )
    input_group.add_argument(
        "--dataset-json",
        type=Path,
        help="Dataset JSON containing an ordered two-entry 'img' list.",
    )
    input_group.add_argument(
        "--sample-id",
        help="Anonymous ID from data/samples.json, for example sample_000.",
    )
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--record-index", type=int, default=0)

    parser.add_argument(
        "--parts-manifest",
        type=Path,
        help="Optional two-part checkpoint manifest override.",
    )
    parser.add_argument("--llama-model", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--prompt")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--num-beams", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--do-sample", action="store_true", default=None)
    parser.add_argument("--output", type=Path, help="Optional plain-text report file.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly allow replacing an existing --output file.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


# 将三种输入方式收口为同一有序路径对，公开样本和本地记录的报告、caption 与 label 均不传入模型。
def _resolve_image_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.images is not None:
        return tuple(Path(path).expanduser().resolve() for path in args.images)  # type: ignore[return-value]
    if args.sample_id is not None:
        return load_public_sample(args.sample_id, args.manifest).image_paths
    if args.image_root is None:
        raise ValueError("--image-root is required with --dataset-json")
    return resolve_pair_from_dataset_record(
        args.dataset_json,
        image_root=args.image_root,
        split=args.split,
        record_index=args.record_index,
    )


# 新增安全文本输出，默认拒绝覆盖已有结果，并保证 stdout 只包含最终报告。
def _write_report(report: str, output: Path | None, *, force: bool) -> None:
    if output is None:
        print(report)
        return
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output}. Pass --force explicitly."
        )
    mode = "w" if force else "x"
    with output.open(mode, encoding="utf-8") as handle:
        handle.write(report.rstrip() + "\n")
    print(report)


# CLI 主流程只覆盖并校验二分片清单，再执行双图报告生成。
def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    settings = ReleaseSettings.from_yaml(args.config).with_overrides(
        parts_manifest=args.parts_manifest,
        llama_model=args.llama_model,
        device=args.device,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        top_p=args.top_p,
        temperature=args.temperature,
        do_sample=args.do_sample,
    )
    image_paths = _resolve_image_paths(args)
    pipeline = ReportGenerationPipeline(settings)
    report = pipeline.generate(image_paths)
    _write_report(report, args.output, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
