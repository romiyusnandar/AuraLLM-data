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
  Main translation script for batch translation by row range.
- `merge_filter_jsonl.py`
  Merge translated shards and drop clearly bad records.
- `normalize_jsonl_labels.py`
  Normalize `Human:` / `Assistant:` labels into Indonesian role labels.
- `audit_jsonl_quality.py`
  Audit output quality before you use the dataset for training.

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

## Recommended Workflow

1. Translate data in chunks.
2. Merge all chunk files into one file.
3. Filter obviously bad records such as empty rows or equal pairs.
4. Normalize speaker labels.
5. Audit the final file before training.

Example:

```bash
python translate_fast.py --start 0 --end 10000 --output part_00000_10000.jsonl
python translate_fast.py --start 10000 --end 20000 --output part_10000_20000.jsonl
python merge_filter_jsonl.py --input "part_*.jsonl" --output final_merged.jsonl --drop-equal --drop-empty --dedup-by-idx
python normalize_jsonl_labels.py --input final_merged.jsonl --output final_normalized.jsonl
python audit_jsonl_quality.py --input final_normalized.jsonl
```

## Colab Notes

Typical Google Colab flow:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!git clone https://github.com/romiyusnandar/AuraLLM-data.git
%cd AuraLLM-data
!pip install -r requirements.txt
!python translate_fast.py --start 0 --end 1000 --output part_000_100.jsonl --drive
```

If needed, you can then merge and clean output files stored in Google Drive:

```bash
!python merge_filter_jsonl.py --input "/content/drive/MyDrive/AuraLLM-data/part_*.jsonl" --output "/content/drive/MyDrive/AuraLLM-data/final_merged.jsonl" --drop-equal --drop-empty --dedup-by-idx
!python normalize_jsonl_labels.py --input "/content/drive/MyDrive/AuraLLM-data/final_merged.jsonl" --output "/content/drive/MyDrive/AuraLLM-data/final_normalized.jsonl"
!python audit_jsonl_quality.py --input "/content/drive/MyDrive/AuraLLM-data/final_normalized.jsonl"
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
