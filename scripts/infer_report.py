#!/usr/bin/env python3
"""Source-tree entry point for report generation."""

from pathlib import Path
import sys

# 新增无需预先安装 package 的推理入口，仅将当前发布包 src 目录加入搜索路径。
RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT / "src"))

from mam_wisequery_report.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

