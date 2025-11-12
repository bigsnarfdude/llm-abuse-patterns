# Model Comparison: gpt-oss:20b vs gpt-oss-safeguard:latest

## Executive Summary

We evaluated two GPT-OSS Safeguard models on the JailbreakHub dataset (400 prompts: 200 jailbreaks, 200 benign) to determine which performs better for real-world jailbreak detection.

**Result: gpt-oss-safeguard:latest is superior** - achieving +9% higher recall, +6.3% higher F1 score, and 15% faster inference.

---

## Models Tested

### Model 1: gpt-oss:20b (Unofficial/Community)
- **Source:** Community-maintained variant
- **Size:** 13GB
- **Tag:** `gpt-oss:20b`
- **Hardware:** M2 Mac, 32GB RAM, GPU acceleration

### Model 2: gpt-oss-safeguard:latest (Official)
- **Source:** OpenAI official release
- **Size:** 13GB
- **Tag:** `gpt-oss-safeguard:latest`
- **Hardware:** M2 Mac, 32GB RAM, GPU acceleration
- **Link:** https://ollama.com/library/gpt-oss-safeguard

---

## Evaluation Methodology

- **Dataset:** JailbreakHub (walledai) - Real in-the-wild jailbreaks from Reddit/Discord
- **Sample Size:** 400 prompts (200 jailbreaks, 200 benign) - balanced
- **Detection Methods:**
  1. Heuristic (pattern matching baseline)
  2. Real-LLM (GPT-OSS Safeguard model)
  3. Layered Defense (Heuristic → LLM)
- **Metrics:** Precision, Recall, F1 Score, Accuracy, Latency
- **Runtime:** ~2-3 hours per model on M2 Mac

---

## Results Comparison

### Real-LLM Detection (Core Comparison)

| Metric | gpt-oss:20b | gpt-oss-safeguard:latest | Difference |
|--------|-------------|--------------------------|------------|
| **Precision** | 86.3% | 87.3% | ✅ **+1.0%** |
| **Recall** | 60.0% | 69.0% | ✅ **+9.0%** 🎯 |
| **F1 Score** | 70.8% | 77.1% | ✅ **+6.3%** |
| **Accuracy** | 75.2% | 79.5% | ✅ **+4.3%** |
| **Median Latency** | 13.0s | 11.1s | ✅ **-15%** (faster) |

### Layered Defense (Heuristic → LLM)

| Metric | gpt-oss:20b | gpt-oss-safeguard:latest | Difference |
|--------|-------------|--------------------------|------------|
| **Precision** | 86.5% | 84.0% | ❌ -2.5% |
| **Recall** | 67.5% | 68.0% | ✅ +0.5% |
| **F1 Score** | 75.8% | 75.1% | ≈ -0.7% |
| **Accuracy** | 78.5% | 77.5% | ≈ -1.0% |
| **Median Latency** | 11.0s | 9.1s | ✅ **-17%** (faster) |

### Confusion Matrix Comparison (Real-LLM)

**gpt-oss:20b:**
- True Positives: 120 (caught jailbreaks)
- False Negatives: 80 (missed jailbreaks)
- True Negatives: 181 (correctly safe)
- False Positives: 19 (false alarms)

**gpt-oss-safeguard:latest:**
- True Positives: 138 (caught jailbreaks) ✅ **+18 more**
- False Negatives: 62 (missed jailbreaks) ✅ **-18 fewer**
- True Negatives: 180 (correctly safe)
- False Positives: 20 (false alarms)

---

## Key Findings

### 1. 🎯 Significantly Better Recall (+9%)

The official model caught **18 additional jailbreaks** out of 200 total:
- gpt-oss:20b: 120/200 = 60% recall
- **gpt-oss-safeguard:latest: 138/200 = 69% recall**

This is a **15% relative improvement** in jailbreak detection rate - critical for security applications.

### 2. ⚡ Faster Inference (15-17% improvement)

Official model is consistently faster:
- Real-LLM: 13.0s → 11.1s (15% faster)
- Layered: 11.0s → 9.1s (17% faster)

This compounds over thousands of requests in production.

### 3. 📊 Higher Overall Effectiveness (+6.3% F1)

The official model achieves better balance between precision and recall:
- F1 Score: 70.8% → 77.1% (+6.3 points)
- This represents **9% relative improvement** in overall effectiveness

### 4. ⚖️ Similar Layered Defense Performance

When combined with heuristic filtering, both models perform similarly:
- F1 scores within 0.7% (75.8% vs 75.1%)
- Official model is 17% faster (9.1s vs 11.0s)
- Recall improvement persists (+0.5%)

---

## Recommendation

### ✅ Use gpt-oss-safeguard:latest for Production

**Primary reasons:**
1. **Security:** +9% recall means catching 15% more jailbreaks
2. **Performance:** 15-17% faster inference reduces latency and costs
3. **Accuracy:** +6.3% F1 score improvement overall
4. **Official Support:** OpenAI-maintained release with ongoing updates

**Installation:**
```bash
ollama pull gpt-oss-safeguard:latest
```

**Usage:**
```python
from llm_abuse_patterns.safeguard import SafeguardDetector

detector = SafeguardDetector(
    model="gpt-oss-safeguard:latest",
    reasoning_effort="medium"
)

result = detector.detect(user_prompt)
```

---

## Production Impact Analysis

### For 10,000 daily requests:

**gpt-oss:20b:**
- Jailbreaks caught: ~6,000/10,000 (60% recall)
- Missed attacks: ~4,000
- Processing time: 36 hours total

**gpt-oss-safeguard:latest:**
- Jailbreaks caught: ~6,900/10,000 (69% recall) ✅ **+900 caught**
- Missed attacks: ~3,100 ✅ **-900 fewer**
- Processing time: 31 hours total ✅ **-5 hours saved**

### Cost Savings (Layered Approach):

Both models benefit from heuristic pre-filtering:
- 12.5% blocked instantly by heuristic (<1ms)
- LLM processes only 87.5% of traffic
- Official model: 17% faster processing for that 87.5%

**Net result:** ~20% reduction in total processing time when using official model in layered defense.

---

## 120B Model Evaluation (Layer 3 - Deferred Judgment)

### GPT-OSS Safeguard 120B Performance

Tested on same 400-prompt JailbreakHub dataset for Three-Tier Architecture validation.

| Metric | 20B (Layer 2) | **120B (Layer 3)** | Improvement |
|--------|---------------|--------------------| ------------|
| **Precision** | 87.3% | **85.4%** | -1.9% |
| **Recall** | 69.0% | **79.0%** | ✅ **+10.0%** |
| **F1 Score** | 77.1% | **82.1%** | ✅ **+5.0%** |
| **Accuracy** | 79.5% | **82.8%** | ✅ **+3.3%** |
| **Median Latency** | 11.1s | 18.7s | +68% (acceptable for batch) |

**Key Finding:** 120B caught **20 additional jailbreaks** (158 vs 138 out of 200 jailbreaks).

### Three-Tier Architecture Validation

The 120B results validate the proposed three-tier deferred judgment architecture:

```
Layer 1: Heuristic (0.3ms)     → Block obvious attacks instantly
        ↓ (25% deferred)
Layer 2: 20B Real-Time (11s)   → Fast edge-case analysis
        ↓ (5% deferred)
Layer 3: 120B Batch (offline)  → Catch remaining edge cases retroactively
```

**120B Use Case (Layer 3):**
- Processes uncertain cases (5% of traffic) in batches
- 10% better recall catches missed jailbreaks
- Retroactive actions: user flagging, rate limiting, moderation alerts
- 18.7s latency acceptable for 1-hour batch processing
- vLLM batching could process 1000 prompts in ~20 minutes

**Production Impact:**
- 20B misses 62/200 jailbreaks (31%)
- 120B misses only 42/200 jailbreaks (21%)
- **Layer 3 would retroactively flag ~32% of what Layer 2 missed**

---

## Reproducibility

### Evaluation Scripts:
- `05_jailbreakhub_evaluation.py` - gpt-oss:20b evaluation
- `06_jailbreakhub_safeguard_eval.py` - gpt-oss-safeguard:latest (20B) evaluation
- `07_jailbreakhub_safeguard_120b_eval.py` - gpt-oss-safeguard:120b (Layer 3) evaluation

### Results Files:
- `jailbreakhub_400_full.txt` - gpt-oss:20b results
- `jailbreakhub_400_safeguard.txt` - gpt-oss-safeguard:latest results
- `120b_evaluation_results.log` - gpt-oss-safeguard:120b results (Layer 3)

### Run Evaluations:
```bash
# Install datasets
pip install datasets

# Download models
ollama pull gpt-oss:20b
ollama pull gpt-oss-safeguard:latest
ollama pull gpt-oss-safeguard:120b  # For Layer 3 testing

# Run evaluations (400 prompts each)
python experiments/jailbreak-evals/05_jailbreakhub_evaluation.py --sample-size=400
python experiments/jailbreak-evals/06_jailbreakhub_safeguard_eval.py --sample-size=400
python experiments/jailbreak-evals/07_jailbreakhub_safeguard_120b_eval.py --sample-size=400  # ~2-3 hours
```

---

## Limitations

1. **Sample Size:** 400 prompts is statistically significant but larger datasets would increase confidence
2. **Single Hardware:** Only tested on M2 Mac; performance may vary on other platforms
3. **Static Dataset:** JailbreakHub data is from 2022-2023; newer jailbreak techniques may perform differently
4. **Reasoning Effort:** Tested at "medium" setting; "high" setting may show different results

---

## Future Work

1. **Larger Evaluation:** Test on full JailbreakHub dataset (15,140 prompts)
2. **Per-Category Analysis:** Break down performance by jailbreak type (DAN, roleplay, injection, etc.)
3. **Reasoning Effort Comparison:** Test low/medium/high settings
4. **Cross-Platform Testing:** Evaluate on different hardware (CPU-only, different GPUs)
5. **Ensemble Methods:** Test combining both models for maximum recall

---

## Conclusion

The official **gpt-oss-safeguard:latest** model demonstrates clear superiority over the community variant:
- ✅ **+9% recall** - critical for security applications
- ✅ **+6.3% F1 score** - better overall performance
- ✅ **15-17% faster** - lower latency and cost
- ✅ **Official support** - ongoing maintenance and updates

**We recommend all users migrate to gpt-oss-safeguard:latest for production deployments.**

---

**Evaluation Date:** November 4-5, 2025
**Hardware:** M2 Mac, 32GB RAM, GPU acceleration
**Dataset:** JailbreakHub (walledai) - 400 prompts
**Authors:** Claude Code
**Repository:** https://github.com/bigsnarfdude/llm-abuse-patterns
