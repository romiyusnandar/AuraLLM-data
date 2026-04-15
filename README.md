# AuraLLM Data Tools

Translation utilities for preparing Indonesian training data for AuraLLM.

## What This Repo Actually Contains

This repository currently contains a small set of Python scripts for:

- downloading a Hugging Face dataset
- translating English text to Indonesian with `Helsinki-NLP/opus-mt-en-id`
- splitting work by shard or row range
- saving translated output as JSONL
- resuming interrupted runs

The current scripts are focused on the `Anthropic/hh-rlhf` dataset and its
`chosen` / `rejected` fields.

## Included Scripts

- `translate_fast.py`
  Best starting point for fast batch translation by row range.
- `translate_range.py`
  Simpler range-based translation flow.
- `translate_v2.py`
  Shard-based batch translation flow.
- `translate_pipeline.py`
  Older distributed script kept for reference. Review before using in public or
  shared environments.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick Start

Translate rows `0..999` from `Anthropic/hh-rlhf` into `hasil_0_1000.jsonl`:

```bash
python translate_fast.py --start 0 --end 1000 --output hasil_0_1000.jsonl
```

Example output format:

```json
{"chosen":"...", "rejected":"...", "original_idx":123}
```

## Notes Before Publishing

- This repo contains code only. Check the license of any source dataset before
  publishing generated data.
- Do not commit generated `.jsonl` files, secrets, or local caches.
- Some scripts include Google Colab / Google Drive-oriented paths.
- The scripts are still opinionated toward one dataset and are not yet fully
  generic.

## Suggested Next Step

If you want to adopt this repo for your own datasets, the best path is to make
`translate_fast.py` configurable for:

- dataset name
- split name
- input text columns
- output file schema

## License

Apache License 2.0. See [LICENSE](LICENSE).
