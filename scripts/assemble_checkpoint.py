#!/usr/bin/env python3
"""Verify and assemble the two GitHub Release checkpoint parts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


RELEASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = RELEASE_ROOT / "weights" / "release_parts.json"
CHUNK_SIZE = 16 * 1024 * 1024


# 新增流式文件及有序分片 SHA-256 计算，校验时不需要把 2.3 GiB 权重整体读入内存。
def _sha256_paths(paths: Iterable[Path]) -> tuple[int, str]:
    digest = hashlib.sha256()
    total_size = 0
    for path in paths:
        with path.open("rb") as handle:
            while chunk := handle.read(CHUNK_SIZE):
                digest.update(chunk)
                total_size += len(chunk)
    return total_size, digest.hexdigest()


# 将清单中的文件名限制在 weights 清单目录内，避免通过相对路径读取或写入目录外文件。
def _resolve_artifact(root: Path, filename: object) -> Path:
    if not isinstance(filename, str) or not filename.strip():
        raise TypeError("Release artifact filename must be a non-empty string")
    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Release artifact escapes its directory: {filename}") from exc
    return candidate


# 严格读取版本 1 的二分片发布清单，并以兼容 Python 3.10 的显式列表构造校验全部资产元数据。
def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("format") != "mam-wisequery-report-release-parts":
        raise ValueError("Unsupported checkpoint release-parts manifest format")
    if payload.get("version") != 1 or payload.get("part_count") != 2:
        raise ValueError("Only version 1 two-part checkpoint releases are supported")
    parts = payload.get("parts")
    assembled = payload.get("assembled")
    if not isinstance(parts, list) or len(parts) != 2:
        raise ValueError("The checkpoint release manifest must contain exactly two parts")
    if not isinstance(assembled, dict):
        raise TypeError("The checkpoint release manifest must contain 'assembled'")
    artifacts = [(f"part {index}", part) for index, part in enumerate(parts)]
    artifacts.append(("assembled checkpoint", assembled))
    for label, artifact in artifacts:
        if not isinstance(artifact, dict):
            raise TypeError(f"{label} metadata must be an object")
        if not isinstance(artifact.get("size_bytes"), int):
            raise TypeError(f"{label} size_bytes must be an integer")
        digest = artifact.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{label} sha256 must contain 64 hexadecimal characters")
        int(digest, 16)
    return payload


# 逐个检查 Release 分片的存在性、大小和哈希，并再次校验按清单顺序拼接后的最终哈希。
def _verify_parts(
    manifest_path: Path,
    payload: dict[str, Any],
) -> tuple[list[Path], dict[str, Any]]:
    artifact_root = manifest_path.parent
    part_paths: list[Path] = []
    for index, artifact in enumerate(payload["parts"]):
        path = _resolve_artifact(artifact_root, artifact.get("filename"))
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint part is missing: {path}")
        size, digest = _sha256_paths([path])
        if size != artifact["size_bytes"] or digest != artifact["sha256"]:
            raise RuntimeError(
                f"Checkpoint part {index} failed verification: {path.name}"
            )
        print(f"verified part {index + 1}/2: {path.name}")
        part_paths.append(path)

    assembled = payload["assembled"]
    combined_size, combined_digest = _sha256_paths(part_paths)
    if (
        combined_size != assembled["size_bytes"]
        or combined_digest != assembled["sha256"]
    ):
        raise RuntimeError("Ordered checkpoint parts do not match the final checkpoint")
    print("verified ordered two-part checkpoint payload")
    return part_paths, assembled


# 采用临时文件原子生成最终 checkpoint，拒绝将输出指向输入分片，并默认拒绝覆盖既有权重。
def assemble_checkpoint(
    manifest_path: str | Path,
    *,
    output: str | Path | None = None,
    force: bool = False,
    verify_only: bool = False,
) -> Path | None:
    manifest_path = Path(manifest_path).expanduser().resolve()
    payload = _load_manifest(manifest_path)
    part_paths, assembled = _verify_parts(manifest_path, payload)
    if verify_only:
        return None

    output_path = (
        Path(output).expanduser().resolve()
        if output is not None
        else _resolve_artifact(manifest_path.parent, assembled.get("filename"))
    )
    if output_path in part_paths:
        raise ValueError("The assembled checkpoint output cannot replace an input part")
    if output_path.exists():
        size, digest = _sha256_paths([output_path])
        if size == assembled["size_bytes"] and digest == assembled["sha256"]:
            print(f"assembled checkpoint is already valid: {output_path}")
            return output_path
        if not force:
            raise FileExistsError(
                f"Refusing to replace invalid existing checkpoint: {output_path}. "
                "Pass --force explicitly to replace it."
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".assembling")
    if temporary_path.exists():
        raise FileExistsError(f"Temporary assembly file already exists: {temporary_path}")

    try:
        with temporary_path.open("xb") as destination:
            for part_path in part_paths:
                with part_path.open("rb") as source:
                    while chunk := source.read(CHUNK_SIZE):
                        destination.write(chunk)
        size, digest = _sha256_paths([temporary_path])
        if size != assembled["size_bytes"] or digest != assembled["sha256"]:
            raise RuntimeError("Assembled checkpoint failed final verification")
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    print(f"assembled checkpoint: {output_path}")
    return output_path


# 提供默认读取 weights/release_parts.json 的跨平台命令行入口，并显式控制校验、输出与覆盖行为。
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    assemble_checkpoint(
        args.manifest,
        output=args.output,
        force=args.force,
        verify_only=args.verify_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
