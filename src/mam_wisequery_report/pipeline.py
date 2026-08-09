"""End-to-end ordered two-image to OCT report pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from .model_loader import ModelLoadAudit, load_report_model
from .preprocessing import OCTImagePreprocessor, load_ordered_image_pair
from .settings import ReleaseSettings


# 新增从有序两图到单个文本报告的完整推理管线，公开接口不返回分类 logits、概率或标签。
class ReportGenerationPipeline:
    def __init__(
        self,
        settings: ReleaseSettings,
        *,
        model=None,
        load_audit: ModelLoadAudit | None = None,
    ):
        self.settings = settings
        if model is None:
            model, load_audit = load_report_model(settings)
        self.model = model
        self.load_audit = load_audit
        self.preprocessor = OCTImagePreprocessor(settings.preprocess)
        self.device = torch.device(settings.runtime.device)

    @torch.inference_mode()
    def generate(self, image_paths: Sequence[str | Path]) -> str:
        images = load_ordered_image_pair(image_paths, self.preprocessor).to(self.device)
        generation = self.settings.generation
        reports = self.model.generate(
            images=images,
            texts=[generation.prompt],
            num_beams=generation.num_beams,
            max_new_tokens=generation.max_new_tokens,
            min_length=generation.min_length,
            top_p=generation.top_p,
            temperature=generation.temperature,
            do_sample=generation.do_sample,
        )
        if not isinstance(reports, list) or len(reports) != 1:
            raise RuntimeError(
                f"Expected exactly one generated report, got {type(reports).__name__}: {reports}"
            )
        return str(reports[0]).strip()

    @torch.inference_mode()
    def encode_signature(
        self, image_paths: Sequence[str | Path]
    ) -> dict[str, torch.Tensor]:
        images = load_ordered_image_pair(image_paths, self.preprocessor).to(self.device)
        llm_embeds, llm_attention, qformer_output = self.model.encode_img(images)
        return {
            "llm_embeds": llm_embeds.detach().float().cpu(),
            "llm_attention": llm_attention.detach().cpu(),
            "qformer_output": qformer_output.detach().float().cpu(),
        }

