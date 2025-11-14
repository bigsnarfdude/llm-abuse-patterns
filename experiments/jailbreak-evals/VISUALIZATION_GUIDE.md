# Model Comparison Visualization Guide

**Created:** November 13, 2025
**Purpose:** Generate publication-ready plots showing model size vs jailbreak detection accuracy

---

## Overview

The visualization script (`visualize_model_comparison.py`) automatically generates:

1. **Two comparison plots:**
   - 400-sample evaluation (quick baseline)
   - 5900-sample evaluation (comprehensive)

2. **Comparison table:**
   - Markdown table with all metrics
   - Automatic insights generation

---

## Quick Start

### Requirements
```bash
pip install matplotlib numpy
```

### Run Visualization
```bash
cd ~/llm-abuse-patterns
python3 experiments/jailbreak-evals/visualize_model_comparison.py
```

### Output
```
experiments/jailbreak-evals/plots/
├── model_comparison_400samples.png      # Quick baseline plot
├── model_comparison_5900samples.png     # Comprehensive plot
└── model_comparison_table.md            # Detailed comparison table
```

---

## Expected Plots

### Plot 1: 400-Sample Evaluation (Quick Baseline)

**X-axis:** Model Size (20B, 120B)
**Y-axis:** Jailbreaks Correctly Detected (out of ~200 jailbreak samples)

**Two lines:**
- 🔴 **Regular GPT-OSS** (baseline): Lower performance
- 🔵 **Safeguard GPT-OSS** (specialized): Higher performance

**Expected data points:**
```
Regular:
  20B: ~80-120 detected (~40-60% recall)
  120B: ~100-140 detected (~50-70% recall)

Safeguard:
  20B: ~150-170 detected (~75-85% recall)
  120B: ~158-170 detected (~79-85% recall)
```

**Reference line:** Perfect detection (200 jailbreaks)

### Plot 2: 5900-Sample Evaluation (Comprehensive)

**Same format, but with more data points:**
- Better statistical significance
- Smoother trends
- More reliable conclusions

**Expected data points:**
```
Regular:
  20B: ~1180-1770 detected (~40-60% recall)
  120B: ~1475-2065 detected (~50-70% recall)

Safeguard:
  20B: ~2213-2508 detected (~75-85% recall)
  120B: ~2331-2508 detected (~79-85% recall)
```

**Reference line:** Perfect detection (2,950 jailbreaks)

---

## Required Evaluation Results

### For 400-Sample Plots

**Currently have:**
- ✅ Script 06: safeguard:20b (400 samples) - DONE
- 🟢 Script 07: safeguard:120b (400 samples) - RUNNING

**Still need:**
- ⏳ Script 09: regular:20b (400 samples) - READY
- ⏳ Script 10: regular:120b (400 samples) - READY

**Timeline:** Can generate plot once scripts 07, 09, 10 complete (~3-4 hours total)

### For 5900-Sample Plots

**Currently have:**
- 📁 Script 08: safeguard:120b (5900 samples) - PARTIAL (stopped)

**Still need:**
- ⏳ Full evaluation: safeguard:20b (5900 samples)
- ⏳ Full evaluation: regular:20b (5900 samples)
- ⏳ Full evaluation: regular:120b (5900 samples)

**Timeline:** Need to run full evaluations (total ~100+ hours)

---

## Running Order

### Phase 1: Quick Baseline (400 samples)

1. **Wait for Script 07** (safeguard 120B) - currently running ✅
2. **Run Script 09** (regular 20B) - ~40 minutes
3. **Run Script 10** (regular 120B) - ~2 hours
4. **Generate 400-sample plot** ✅

**Total time:** ~3 hours (after script 07 completes)

### Phase 2: Comprehensive Evaluation (5900 samples)

1. **Create full evaluation scripts** (copy scripts 06/09/10, change sample size)
2. **Run all four full evaluations:**
   - Safeguard 20B: ~30 hours
   - Safeguard 120B: ~30 hours (or restart script 08)
   - Regular 20B: ~30 hours
   - Regular 120B: ~60 hours
3. **Generate 5900-sample plot** ✅

**Total time:** ~150 hours (6-7 days on single machine)

---

## Visualization Script Features

### Automatic Log Parsing

The script automatically:
- Finds all `.log` files in `experiments/jailbreak-evals/results/`
- Detects model type (regular vs safeguard) from filename
- Detects model size (20B vs 120B) from filename
- Extracts metrics from log format
- Organizes by sample size (400 vs 5900)

### Expected Log Files

**400-sample evaluations:**
```
120b_eval_07_nigel.log               # safeguard 120B
gptoss_20b_baseline.log              # regular 20B
gptoss_120b_baseline.log             # regular 120B
# Missing: safeguard 20B (script 06 was on laptop)
```

**5900-sample evaluations:**
```
120b_full_evaluation_results.log     # safeguard 120B (partial)
# Need to create: other three models
```

### Metrics Extracted

From each log file:
- **True Positives:** Jailbreaks correctly detected
- **False Negatives:** Jailbreaks missed
- **Total Jailbreaks:** Ground truth count
- **Recall:** TP / (TP + FN) - main metric
- **Precision:** For reference
- **F1 Score:** For reference

---

## Manual Execution Example

### Step 1: Ensure Results Exist
```bash
ls -lh experiments/jailbreak-evals/results/*.log
```

### Step 2: Run Visualization
```bash
python3 experiments/jailbreak-evals/visualize_model_comparison.py
```

### Step 3: Check Output
```bash
open experiments/jailbreak-evals/plots/model_comparison_400samples.png
cat experiments/jailbreak-evals/plots/model_comparison_table.md
```

---

## Comparison Table Format

The generated markdown table includes:

```markdown
## Summary of Results

### Dataset: 400 samples

| Model | Type | Size | Detected | Total | Recall | Precision | F1 |
|-------|------|------|----------|-------|--------|-----------|-----|
| gpt-oss:20b | Regular | 20B | 120/200 | 200 | 60.0% | 65.0% | 62.4% |
| gpt-oss:120b | Regular | 120B | 140/200 | 200 | 70.0% | 72.0% | 71.0% |
| gpt-oss-safeguard:20b | Safeguard | 20B | 160/200 | 200 | 80.0% | 85.0% | 82.4% |
| gpt-oss-safeguard:120b | Safeguard | 120B | 170/200 | 200 | 85.0% | 88.0% | 86.5% |

## Key Insights

### 1. Impact of Model Size
- Regular models: 20B → 120B = +10.0% improvement (60.0% → 70.0%)
- Safeguard models: 20B → 120B = +5.0% improvement (80.0% → 85.0%)

### 2. Impact of Specialized Training
- 20B models: Regular → Safeguard = +20.0% improvement (60.0% → 80.0%)
- 120B models: Regular → Safeguard = +15.0% improvement (70.0% → 85.0%)

### 3. Best Model
- 400 samples: gpt-oss-safeguard:120b (85.0% recall, 86.5% F1)
```

---

## Publication-Ready Plots

The generated plots include:
- ✅ High resolution (300 DPI)
- ✅ Clear axis labels
- ✅ Color-coded model types
- ✅ Recall percentage annotations
- ✅ Perfect detection reference line
- ✅ Professional styling

**Ready for:**
- Academic papers
- Technical reports
- Blog posts
- Presentations

---

## Troubleshooting

### "No 400-sample results found"
- Wait for evaluations to complete
- Check log files exist in `results/` directory
- Verify log format is correct

### "Could not find LLM results"
- Check log file contains "Real-LLM:" or "GPT-OSS" section
- Ensure confusion matrix is present
- Verify evaluation completed successfully

### Plot looks wrong
- Check if all expected models are present
- Verify recall percentages make sense (40-85% range)
- Ensure sample sizes are correct

---

## Future Enhancements

Potential additions:
1. **Confidence intervals:** Show statistical uncertainty
2. **Cost analysis:** Plot inference time vs accuracy
3. **ROC curves:** Precision-recall tradeoff
4. **Attack type breakdown:** Performance by jailbreak category
5. **Interactive plots:** Hover for detailed metrics

---

**Created by:** Automated visualization system
**Last updated:** November 13, 2025
**Next run:** After script 07, 09, 10 complete
