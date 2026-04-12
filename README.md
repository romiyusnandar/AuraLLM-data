# 📊 aksara-data

**Data curation pipeline for aksaraLLM — 100% transparent, 100% reproducible.**

<p align="center">
  <a href="https://github.com/aksaraLLM/aksara-data/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://discord.gg/aksarallm"><img src="https://img.shields.io/badge/Discord-Join-7289da?logo=discord" alt="Discord"></a>
</p>

---

## Overview

This repository contains the complete data pipeline for aksaraLLM pre-training and fine-tuning datasets. Everything is open and documented — from raw source URLs to the final tokenized data.

### What's Included
- 📥 **Download scripts** for all data sources
- 🔍 **Quality filtering** (perplexity, heuristic, classifier-based)
- 🧹 **Deduplication** (MinHash, exact dedup)
- 🔒 **PII removal** pipeline
- 🌐 **Language detection** & filtering
- 📊 **Data analysis** & statistics tools
- 🧪 **Data mixing** configuration

## Data Sources

### Pre-Training Data

| Source | Languages | Tokens (Est.) | License | Status |
|--------|-----------|---------------|---------|--------|
| Common Crawl (filtered) | Multilingual | ~500B | CC-BY-SA | 📋 Planned |
| Wikipedia | ID, EN, +15 langs | ~20B | CC-BY-SA | 📋 Planned |
| CulturaX | Multilingual | ~200B (sampled) | Various | 📋 Planned |
| RedPajama-V2 | EN-heavy | ~300B (sampled) | Apache 2.0 | 📋 Planned |
| arXiv | EN | ~50B | Various | 📋 Planned |
| GitHub (permissive) | Code | ~100B | Permissive | 📋 Planned |
| Stack Exchange | EN | ~15B | CC-BY-SA | 📋 Planned |
| Indonesian Web Crawl | ID | ~50B | Custom | 📋 Planned |
| Books (public domain) | Multi | ~20B | Public Domain | 📋 Planned |

### Fine-Tuning Data

| Dataset | Type | Size | Language |
|---------|------|------|----------|
| OpenHermes 2.5 | Instruction | ~1M examples | EN |
| SlimOrca | Instruction | ~500K examples | EN |
| Indonesian Instructions | Instruction | TBD | ID |
| Code Instructions | Code SFT | TBD | Code |
| UltraFeedback | Preference | ~64K examples | EN |

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE                           │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Download  │→ │ Language  │→ │ Quality  │→ │  Dedup   │   │
│  │ & Extract │  │ Detect   │  │ Filter   │  │ (MinHash)│   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│                                       ↓                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │ Tokenize │← │   Mix    │← │   PII    │                  │
│  │ & Pack   │  │ & Sample │  │ Removal  │                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
│       ↓                                                     │
│  ┌──────────┐                                               │
│  │  Final   │ → Ready for training                          │
│  │ Dataset  │                                               │
│  └──────────┘                                               │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/aksaraLLM/aksara-data.git
cd aksara-data

# Install dependencies
pip install -e ".[dev]"

# Download a specific data source
python -m aksara_data.download --source wikipedia --languages id,en

# Run quality filtering
python -m aksara_data.filter --input data/raw/ --output data/filtered/

# Run deduplication
python -m aksara_data.dedup --input data/filtered/ --output data/deduped/

# Generate dataset statistics
python -m aksara_data.stats --input data/deduped/
```

## Project Structure

```
aksara-data/
├── aksara_data/
│   ├── __init__.py
│   ├── download/              # Data source downloaders
│   │   ├── common_crawl.py
│   │   ├── wikipedia.py
│   │   ├── culturax.py
│   │   └── ...
│   ├── filter/                # Quality filtering
│   │   ├── perplexity.py
│   │   ├── heuristic.py
│   │   ├── classifier.py
│   │   └── language_detect.py
│   ├── dedup/                 # Deduplication
│   │   ├── minhash.py
│   │   ├── exact_dedup.py
│   │   └── url_dedup.py
│   ├── privacy/               # PII removal
│   │   ├── pii_detector.py
│   │   └── anonymizer.py
│   ├── mix/                   # Data mixing
│   │   ├── sampler.py
│   │   └── mixer.py
│   ├── stats/                 # Data analysis
│   │   └── analyzer.py
│   └── utils/
│       └── ...
├── configs/
│   ├── sources.yaml           # Data source definitions
│   ├── filter_config.yaml     # Filtering parameters
│   └── mix_config.yaml        # Data mixing ratios
├── tests/
├── LICENSE
├── README.md
└── pyproject.toml
```

## Data Card

Every dataset we produce comes with a detailed data card documenting:
- **Source**: Where the data came from
- **Processing**: Every transformation applied
- **Statistics**: Token counts, language distribution, quality scores
- **Known Issues**: Any limitations or biases
- **License**: Data licensing information

## Contributing

We especially need help with:
- 🌍 **Non-English data sources** — especially Southeast Asian languages
- 🔍 **Quality filtering improvements**
- 📊 **Data analysis & visualization**
- 🔒 **PII detection** for non-English text

See [CONTRIBUTING.md](https://github.com/aksaraLLM/community/blob/main/CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
