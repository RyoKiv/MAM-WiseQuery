# Model Weights

Complete inference requires:

1. The full [`hfl/chinese-llama-2-7b`](https://huggingface.co/hfl/chinese-llama-2-7b)
   model, downloaded separately. Set `MAM_LLAMA_MODEL` to its local directory.
2. Both checkpoint shards from the
   [`v1.0.0` GitHub Release](https://github.com/RyoKiv/MAM-WiseQuery/releases/tag/v1.0.0).

Download both files into this `weights/` directory. Do not rename, reorder, or
concatenate them:

| Order | Download | Size (bytes) | SHA-256 |
|---:|---|---:|---|
| 1 | [`mam_wisequery_report_inference.pth.part-000`](https://github.com/RyoKiv/MAM-WiseQuery/releases/download/v1.0.0/mam_wisequery_report_inference.pth.part-000) | `1,250,000,000` | `fc7641a263f6072b446b06ca24abe85ad94bebf47369d08c488831ea8e252241` |
| 2 | [`mam_wisequery_report_inference.pth.part-001`](https://github.com/RyoKiv/MAM-WiseQuery/releases/download/v1.0.0/mam_wisequery_report_inference.pth.part-001) | `1,197,917,636` | `c4a9541cb7d406eb28cb6d5344f2890afb03500cc49128a67af788848d859fd4` |

The loader validates and reads both shards directly according to
`release_parts.json`. To verify the downloaded files before inference, run:

```bash
python scripts/assemble_checkpoint.py --verify-only
```
