"""Inference-only utilities for MAM-WiseQuery OCT report generation."""

# 将公开 API 改为延迟导入，查看配置或 checkpoint 时不会提前加载图像/大模型依赖。

__all__ = ["ReleaseSettings", "ReportGenerationPipeline"]
__version__ = "0.1.0"


def __getattr__(name):
    if name == "ReleaseSettings":
        from .settings import ReleaseSettings

        return ReleaseSettings
    if name == "ReportGenerationPipeline":
        from .pipeline import ReportGenerationPipeline

        return ReportGenerationPipeline
    raise AttributeError(name)

