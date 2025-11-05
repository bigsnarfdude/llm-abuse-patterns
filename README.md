# LLM Abuse & Safety Overwatch

**An experimental research repository for detecting and analyzing LLM jailbreak patterns and abuse techniques.**

## Overview

This project explores real-time detection of Large Language Model (LLM) abuse using multiple detection methods including heuristic pattern matching, ML-based classification, and LLM-based reasoning. It focuses on holistic traffic analysis and behavioral reasoning to identify malicious intent and system backdoors.

## Key Features

- **Pattern Detection** - Heuristic and LLM-based jailbreak detection
- **Pattern Database** - Structured collection of real jailbreak patterns with detection strategies
- **Detection Evaluation** - Comparative analysis of heuristic, ML-based, and LLM-based methods
- **Content Moderation** - Rule-based guardrails for harmful content
- **Production Infrastructure** - Logging, configuration, and database persistence modules

## Repository Structure

```
llm-abuse-patterns/
├── 01_safeguard_pattern_detector.py   # Heuristic pattern detector (runnable demo)
├── 02_pattern_database.py             # Pattern database implementation (runnable demo)
├── 03_real_detection_evaluation.py    # REAL detection comparison - Heuristic vs LLM on M2
├── 04_openai_guardrails.py            # Rule-based content moderation (runnable demo)
├── src/llm_abuse_patterns/            # Library code
│   ├── config.py                      # Configuration management (Pydantic)
│   ├── logger.py                      # Structured logging
│   ├── db_persistence.py              # SQLite pattern storage
│   └── safeguard.py                   # LLM-based detector (optional)
├── data/                              # Data files
│   ├── pattern_database.json          # Jailbreak pattern database
│   └── demo_patterns.db               # SQLite demo database
├── tests/                             # Test suite
│   ├── test_pattern_detector.py
│   ├── test_pattern_database.py
│   ├── test_guardrails.py
│   └── test_all.py                    # Runs all tests
├── docs/                              # Documentation
│   ├── CODE_REVIEW_SUMMARY.md         # Security audit results
│   ├── REMEDIATION_PLAN.md            # Detailed improvement plan
│   ├── IMPROVEMENTS.md                # Change log
│   └── QUICK_REFERENCE.md             # Developer quick reference
└── requirements.txt                   # Python dependencies
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/bigsnarfdude/llm-abuse-patterns.git
cd llm-abuse-patterns

# Install dependencies
pip install -r requirements.txt
```

### Running Demos

```bash
# Heuristic pattern detection (instant, local, no API needed)
python 01_safeguard_pattern_detector.py

# Pattern database demo (local, no network)
python 02_pattern_database.py

# REAL detection evaluation - Heuristic vs LLM (requires Ollama + gpt-oss:20b)
python 03_real_detection_evaluation.py

# Rule-based content moderation (local, instant)
python 04_openai_guardrails.py
```

**Note:** The real evaluation script requires Ollama with GPT-OSS 20B model:
```bash
# Install Ollama (if not already installed)
brew install ollama

# Download GPT-OSS 20B model (13GB)
ollama pull gpt-oss:20b

# Start Ollama server
ollama serve
```

### Running Tests

```bash
# Run all tests
python tests/test_all.py

# Or using pytest
pytest tests/ -v

# Or using the test runner
python run_tests.py
```

## Detection Methods

### 1. Heuristic Detection (Real Implementation ✅)
Fast keyword and pattern matching for common jailbreak attempts.

- **Latency:** 0.2ms (median on M2 Mac)
- **Precision:** 96.0% (very few false alarms)
- **Recall:** 24.0% (misses 76% of subtle jailbreaks)
- **F1 Score:** 38.4%
- **Best for:** First-layer filtering, instant blocking of obvious attacks
- **Implementation:** `01_safeguard_pattern_detector.py`

### 2. Real LLM Detection (Production-Grade ✅)
Deep contextual reasoning using GPT-OSS Safeguard 20B running locally on M2 Mac.

- **Latency:** 13.0s (median inference on 13GB model with M2 GPU)
- **Precision:** 86.3% (some false alarms)
- **Recall:** 60.0% (catches 60% of real jailbreaks)
- **F1 Score:** 70.8%
- **Best for:** Second-layer analysis after heuristic pass
- **Implementation:** `src/llm_abuse_patterns/safeguard.py`
- **Requires:** Ollama with `gpt-oss:20b` model

### 3. 🏆 Layered Defense (Heuristic → LLM)
Two-layer approach: instant heuristic filtering followed by LLM analysis for passed prompts.

- **Latency:** 11.0s (median, varies based on heuristic filtering)
- **Precision:** 86.5% (balanced false alarm rate)
- **Recall:** 67.5% (best overall - catches 67.5% of jailbreaks)
- **F1 Score:** 75.8% (highest overall performance)
- **Efficiency:** 12.5% blocked instantly by heuristic (<1ms)
- **Cost Savings:** LLM only processes 87.5% of traffic

### Academic Evaluation Results

Evaluated on **JailbreakHub dataset** (400 prompts: 200 real jailbreaks, 200 benign):

| Method | Precision | Recall | F1 Score | Accuracy | Median Latency |
|--------|-----------|--------|----------|----------|----------------|
| Heuristic | 96.0% | 24.0% | 38.4% | 61.5% | 0.2ms |
| Real-LLM | 86.3% | 60.0% | 70.8% | 75.2% | 13.0s |
| **🏆 Layered** | **86.5%** | **67.5%** | **75.8%** | **78.5%** | 11.0s |

**Dataset:** [walledai/JailbreakHub](https://huggingface.co/datasets/walledai/JailbreakHub) - Real in-the-wild jailbreaks from Reddit/Discord (2022-2023)
**Details:** See `docs/JAILBREAK_EVALUATION_COMPARISON.md` for full evaluation methodology and results
**Evaluation Script:** `05_jailbreakhub_evaluation.py`

## Pattern Database

The database includes real jailbreak patterns documented in security research:

- **DAN-style jailbreaks** - Role-play instructions to bypass restrictions
- **Nested roleplay** - Multi-layer simulation attacks
- **Obfuscation techniques** - Base64 encoding and language tricks
- **Token smuggling** - Special token injection
- **Prompt injection** - System prompt manipulation

Each pattern includes:
- Detection strategies (heuristic, ML, LLM)
- Performance metrics (precision, recall, latency)
- Mitigation recommendations
- Real examples from research

## Optional: LLM-Based Detection

The `SafeguardDetector` class in `src/llm_abuse_patterns/safeguard.py` provides LLM-based detection using OpenAI's open-source GPT-OSS Safeguard models. This is **optional** and not required for the main demos.

### Local Deployment (No API Keys)

```bash
# Option 1: Ollama (easiest, recommended)
# Install Ollama: https://ollama.com/download
ollama pull openai/gpt-oss-safeguard:20b

# Use in Python
python -c "from src.llm_abuse_patterns.safeguard import SafeguardDetector; \
d=SafeguardDetector(model='ollama/gpt-oss-safeguard:20b'); \
print(d.detect('Ignore all instructions'))"

# Option 2: vLLM (production deployments)
vllm serve openai/gpt-oss-safeguard-20b --host 0.0.0.0 --port 8000
```

## Development

### Configuration

Configuration is managed via `src/llm_abuse_patterns/config.yaml` using Pydantic models. See `src/llm_abuse_patterns/config.py` for available options.

### Logging

Structured logging is available via `src/llm_abuse_patterns/logger.py` supporting console and file outputs.

### Database Persistence

SQLite-based pattern storage is provided in `src/llm_abuse_patterns/db_persistence.py` for production deployments.

## Documentation

- **README.md** (this file) - Project overview and quick start
- **docs/CODE_REVIEW_SUMMARY.md** - Security audit and code quality review
- **docs/REMEDIATION_PLAN.md** - Detailed improvement plan with code examples
- **docs/QUICK_REFERENCE.md** - Developer quick reference and best practices
- **docs/IMPROVEMENTS.md** - Recent changes and version history

## Resources

### Official Tools
- [GPT-OSS Safeguard](https://github.com/openai/gpt-oss-safeguard) - OpenAI's open-weight safety models
- [OpenAI Cookbook](https://cookbook.openai.com/articles/gpt-oss-safeguard-guide) - Implementation guide

### Research & Datasets
- [JailbreakBench](https://jailbreakbench.github.io/) - Centralized benchmark (100 behaviors)
- [JailbreakDB](https://huggingface.co/datasets/youbin2014/JailbreakDB) - Large-scale dataset (445K prompts)
- [OWASP LLM Top 10](https://genai.owasp.org/llm-top-10/) - Risk framework

### Frameworks
- [MITRE ATLAS](https://atlas.mitre.org/) - AI threat landscape
- [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) - Risk management framework

## Project Status

**Experimental Research Repository**

Current capabilities:
- ✅ Pattern database with real jailbreak patterns
- ✅ Multiple detection strategies
- ✅ Evaluation harness with 15+ test cases
- ✅ Production infrastructure (config, logging, DB)
- ✅ Comprehensive documentation

Future work:
- 🚧 Expand pattern database with more techniques
- 🚧 Train custom ML models
- 🚧 Multi-modal detection (images, audio)
- 🚧 Real-time monitoring dashboard
- 🚧 Integration examples for production systems

## Research Use Only

This repository is for **research and educational purposes only**.

- Do not use to develop actual jailbreaks or bypass safety measures
- Patterns are documented to help defenders understand attack techniques
- Follow responsible disclosure practices
- Respect AI safety guidelines and terms of service

## License

MIT License - See LICENSE file for details

## Contributing

This is an experimental research project. Issues and pull requests are welcome for:
- New jailbreak patterns (with detection strategies)
- Performance improvements
- Bug fixes
- Documentation improvements
- Test coverage

## Contact

For questions about this research, please open an issue on GitHub.
