# MAM-WiseQuery OCT Report Generation

Minimal inference-only release for generating one Chinese OCT examination
report from an ordered pair of OCT images. The command returns report text
only; it does not return classification labels, probabilities, or logits.

> Research use only. This software is not a medical device and must not be used
> as the sole basis for diagnosis or treatment.

## Associated manuscript

**Disease-Aware Split-Query OCT Report Generation With Query-Conditioned
Visual-Language Alignment**

Kai Wu, Jingtao Wang, Cangxin Li, Qian Cheng, and Xinjian Chen (Member, IEEE).

## Inference path

```text
ordered image pair
  -> RGB conversion and bicubic resize to 224 x 224
  -> RETFound ViT-L/16
  -> Multi-Level Aggregation Module (MAM)
  -> learned image-list position embeddings
  -> split-query Q-Former (WiseQuery)
  -> dynamic QueryNorm prompts
  -> Chinese-LLaMA-2-7B with LoRA
  -> one report string
```

The two input paths are consumed exactly in the supplied order. The interface
does not request, infer, or sort acquisition-view names.

## Repository layout

```text
.
├── configs/inference.yaml       # preprocessing and generation defaults
├── data/                        # 50 anonymous ordered image-pair/report examples
├── scripts/assemble_checkpoint.py # optional integrity check for Release assets
├── scripts/infer_report.py      # public inference entry point
├── src/mam_wisequery_report/    # report-only implementation
├── weights/README.md            # two-part checkpoint download instructions
├── environment.yml
├── requirements.txt
└── README.md
```

## Installation

The reference environment is Python 3.10, PyTorch 2.1.0, CUDA 11.8,
Transformers 4.35.2, PEFT 0.13.2, and a CUDA GPU with at least 40 GiB memory.

Using Conda:

```bash
conda env create -f environment.yml
conda activate mam-wisequery-report
```

Or install the pinned dependencies in an existing CUDA-enabled Python 3.10
environment:

```bash
pip install -r requirements.txt
```

No editable package installation is required. The source-tree entry point adds
`src/` to the Python import path automatically.

## Required weights

Inference requires:

1. The full [`hfl/chinese-llama-2-7b`](https://huggingface.co/hfl/chinese-llama-2-7b)
   base-model directory, downloaded separately from Hugging Face.
2. Both project checkpoint parts listed below.

> **Important:** Complete inference cannot run with the project checkpoint
> alone. This repository and its GitHub Release do not include
> `hfl/chinese-llama-2-7b`; download the full model first and point
> `MAM_LLAMA_MODEL` to its local directory.

The raw training `checkpoint_best.pth` is not a drop-in replacement because it does not contain
all frozen states needed by this standalone inference package.

The GitHub Release distributes the inference checkpoint as two ordered assets:

```text
mam_wisequery_report_inference.pth.part-000
mam_wisequery_report_inference.pth.part-001
```

Download both files into `weights/`. Inference reads them directly as one
virtual seekable stream and does not create a complete `.pth` file. An optional
integrity check is available:

```bash
python scripts/assemble_checkpoint.py --verify-only
```

The default configuration reads `weights/release_parts.json`, verifies both
parts, and passes their virtual concatenation directly to `torch.load`. Expected
properties are recorded in `weights/README.md`. The loader has no complete
checkpoint fallback; `--parts-manifest` can only override the two-part manifest.
Set the external language-model path before inference:

```bash
export MAM_LLAMA_MODEL=/path/to/chinese-llama-2-7b
```

Public inference does not load the original Stage-1 or Stage-2 checkpoints.

## Generate from two OCT images

```bash
python scripts/infer_report.py \
  --images /path/to/image_0.png /path/to/image_1.png
```

To save the report:

```bash
python scripts/infer_report.py \
  --images /path/to/image_0.png /path/to/image_1.png \
  --output outputs/report.txt
```

Existing output files are not replaced unless `--force` is supplied.

## Test with a public example

```bash
python scripts/infer_report.py --sample-id sample_000
```

The command reads the two paths from `data/samples.json`, checks that both files
exist, and preserves their listed order. The reference report is never passed
into the model. These 50 examples demonstrate the interface and are not an
independent benchmark split.

## Optional local dataset record

For a JSON object shaped as
`{split: [{"folder": ..., "img": [first, second], ...}]}`:

```bash
python scripts/infer_report.py \
  --dataset-json /path/to/records.json \
  --image-root /path/to/all_data \
  --split test \
  --record-index 0
```

Only `folder` and the ordered two-entry `img` list are used. Caption and label
fields are ignored.

## Quick validation

```bash
python -m compileall -q src scripts
python scripts/assemble_checkpoint.py --help
python scripts/infer_report.py --help
```

Before publishing the example images, confirm redistribution permission and
add the final GitHub Release page URL to `weights/README.md`.
