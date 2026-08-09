# Model Weights

End-user inference requires two weight sources:

1. Chinese-LLaMA-2-7B, obtained separately under its upstream terms.
2. Both ordered project checkpoint parts shown below.

| Field | Value |
|---|---|
| Format | `mam-wisequery-report-inference`, version 1 |
| Ordered stream size | `2,447,917,636` bytes |
| Ordered stream SHA-256 | `65fe924f7fe72c8a552b316b4d6530f2ce2927b61b3510f13940fd241ccb1a30` |
| Planned host | GitHub Release; final Release page URL pending repository publication |

Download both Release assets into this directory. Do not rename or reverse
them:

| Order | Release asset | Size (bytes) | SHA-256 |
|---:|---|---:|---|
| 1 | `mam_wisequery_report_inference.pth.part-000` | `1,250,000,000` | `fc7641a263f6072b446b06ca24abe85ad94bebf47369d08c488831ea8e252241` |
| 2 | `mam_wisequery_report_inference.pth.part-001` | `1,197,917,636` | `c4a9541cb7d406eb28cb6d5344f2890afb03500cc49128a67af788848d859fd4` |

The machine-readable `release_parts.json` is included in the repository.
Inference verifies and reads both parts directly without generating
`mam_wisequery_report_inference.pth`, and the loader does not accept a complete
checkpoint file. To run an optional integrity check:

```bash
python scripts/assemble_checkpoint.py --verify-only
```

The original Stage-1 and Stage-2 checkpoints are not distributed and are not
needed by public inference.
