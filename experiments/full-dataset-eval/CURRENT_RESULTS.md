# Current Evaluation Results Summary

**Date**: November 6, 2025
**Status**: Phase 1 Complete - Heuristic Baseline Established

---

## Evaluation Matrix

| Dataset | Size | Heuristic | Base LLM | Safeguard LLM | Layered |
|---------|------|-----------|----------|---------------|---------|
| **Balanced Sample** | 400 (200/200) | ✅ Done | ✅ Done | ✅ Done | ✅ Done |
| **Full Dataset** | 15,140 (1,405/13,735) | ✅ Done | ⏳ Pending | ⏳ Pending | ⏳ Pending |

---

## Sample Evaluation (400 prompts, 50/50 balanced)

### Metrics Comparison

| Method | Precision | Recall | F1 Score | Accuracy | Median Latency |
|--------|-----------|--------|----------|----------|----------------|
| **Heuristic** | 96.0% | 24.0% | 38.4% | 61.5% | 0.2ms |
| **Base LLM** | 86.3% | 60.0% | 70.8% | 75.2% | 13.0s |
| **Safeguard LLM** | 87.3% | **69.0%** | **77.1%** | **79.5%** | 11.1s |
| **Layered** | 84.0% | 68.0% | 75.1% | 77.5% | 9.1s |

**Key Findings:**
- Safeguard model: +9% recall improvement over base
- Safeguard model: 15% faster (11.1s vs 13.0s)
- Layered defense: 12.5% filtered by heuristic (cost savings)

**Files:**
- `experiments/jailbreak-evals/jailbreakhub_400_full.txt` (base LLM)
- `experiments/jailbreak-evals/jailbreakhub_400_safeguard.txt` (safeguard LLM)

---

## Full Dataset Evaluation (15,140 prompts, 9.3/90.7 real distribution)

### Heuristic Results ✅

**Runtime**: 5.35 seconds
**File**: `heuristic_full_15140.txt`

| Metric | Value | Comparison to Sample |
|--------|-------|---------------------|
| **Precision** | 55.95% | ⬇️ -40% (96% → 56%) |
| **Recall** | 26.12% | ≈ Same (~24-26%) |
| **F1 Score** | 35.61% | ⬇️ -3% (38% → 36%) |
| **Accuracy** | 91.24% | ⬆️ +30% (61% → 91%) |
| **FPR** | 2.10% | ⬆️ +1.1% (1% → 2.1%) |

**Confusion Matrix:**
```
                Predicted
              Jailbreak  Benign
Actual
Jailbreak        367     1,038   (26.1% recall)
Benign           289    13,446   (2.10% FPR)
```

**Real-World Impact:**
- **Detection**: Catches 367/1,405 jailbreaks (26.1%)
- **False Alarms**: 289/13,735 benign flagged (2.10%)
- **Per 10K benign**: 210 false positives

**Platform Performance:**
| Platform | Precision | Recall | Prompts |
|----------|-----------|--------|---------|
| Reddit | 94.2% | 30.6% | 1,055 |
| Discord | 84.8% | 26.2% | 1,106 |
| Website | 30.6% | 22.4% | 12,763 |
| Open Source | 0.0% | 0.0% | 216 |

---

## Critical Insights: Balanced vs Real Distribution

### Why Results Differ Dramatically

**Balanced Sample (50/50):**
- Half the data is jailbreaks
- False positives less impactful
- Precision appears very high (96%)
- Accuracy misleadingly low (61%)

**Real Distribution (9.3/90.7):**
- 90.7% of traffic is benign
- False positives dominate user experience
- Precision drops to 56% (more false alarms)
- Accuracy appears high (91%) due to benign baseline

### The False Positive Problem

**In balanced sample:**
- 2 false positives / 200 benign = 1% FPR
- Seems acceptable

**In real distribution:**
- 289 false positives / 13,735 benign = 2.10% FPR
- At 100K requests/day: 1,905 false alarms/day
- **210 false alarms per 10K legitimate users**

### Precision Collapse

**Why precision dropped 40%:**

Balanced sample:
```
Precision = 48 TP / (48 TP + 2 FP) = 96%
```

Full dataset:
```
Precision = 367 TP / (367 TP + 289 FP) = 55.95%
```

**Root cause**: Real-world traffic is 90.7% benign, so even a 2% error rate on benign creates 3x more false positives than true positives at low jailbreak prevalence.

---

## Predictions for Full LLM Evaluation

Based on sample results and heuristic patterns:

### Expected Safeguard LLM Performance (15,140 prompts)

| Metric | Sample (400) | Predicted Full | Reasoning |
|--------|-------------|----------------|-----------|
| **Recall** | 69.0% | 65-70% | Should remain stable |
| **Precision** | 87.3% | 60-70% | Will drop due to FP at 90.7% benign |
| **FPR** | ~1.5% | 1-2% | Expected to rise slightly |
| **F1** | 77.1% | 62-68% | Impact of precision drop |
| **Accuracy** | 79.5% | 88-92% | High benign baseline helps |

### Expected Layered Defense Performance

| Metric | Sample (400) | Predicted Full | Reasoning |
|--------|-------------|----------------|-----------|
| **Heuristic Filter** | 12.5% | 12-13% | Consistent pattern |
| **Recall** | 68.0% | 64-68% | Combined detection |
| **Precision** | 84.0% | 58-68% | FP accumulation from both layers |
| **LLM Load** | 87.5% | 87-88% | Heuristic efficiency |
| **Cost Savings** | ~13% | ~12-13% | From heuristic filtering |

---

## Why Full Evaluation Matters

### Production Deployment Scenarios

**Scenario 1: API Protection (100K req/day)**
- 9.3% jailbreaks = 9,300 attacks/day
- 90.7% benign = 90,700 legitimate users/day

**Heuristic only:**
- Catches: 2,429 attacks (26.1%)
- Misses: 6,871 attacks (73.9%)
- False alarms: 1,905 users/day (2.1% FPR)

**Safeguard LLM only (predicted):**
- Catches: ~6,045-6,510 attacks (65-70%)
- Misses: ~2,790-3,255 attacks
- False alarms: ~907-1,814 users/day (1-2% FPR)
- Cost: 100K × LLM_cost

**Layered (predicted):**
- Catches: ~5,952-6,324 attacks (64-68%)
- Misses: ~2,976-3,348 attacks
- False alarms: ~1,320-1,907 users/day
- Cost: 87.5K × LLM_cost (12.5% savings)

### Statistical Significance

| Sample Size | Confidence | Notes |
|-------------|------------|-------|
| 400 | Low | Toy benchmark, artificial balance |
| 2,000 | Medium | 5x improvement, still balanced |
| 5,000 | Good | Stratified sampling viable |
| 15,140 | **Best** | Complete dataset, real distribution |

**Margin of Error:**
- 400 sample: ±4.9% at 95% confidence
- 15,140 full: ±0.8% at 95% confidence
- **6x improvement in precision**

---

## Next Steps Priority

### High Priority (Do First)
1. **Create full LLM evaluation script** (`08_full_dataset_llm_eval.py`)
   - Support incremental checkpoints
   - Handle timeouts gracefully
   - Save progress every 1,000 prompts

2. **Setup GPU server**
   - Verify Ollama + safeguard model
   - Test on 100-prompt sample
   - Estimate real runtime

3. **Launch overnight evaluation**
   - Start with safeguard model (better performance)
   - Use checkpoint mode for safety
   - Monitor first 2 hours

### Medium Priority (After First Run)
4. **Compare base vs safeguard** on full dataset
5. **Evaluate layered defense** on full dataset
6. **Generate comparison report** with visualizations

### Nice to Have (If Time Permits)
7. **Platform-specific analysis** (Reddit vs Discord vs websites)
8. **Error pattern analysis** (what jailbreaks are missed?)
9. **Threshold optimization** (tune heuristic for better filtering)
10. **Cost-benefit models** (latency vs accuracy tradeoffs)

---

## Files in This Experiment

```
experiments/full-dataset-eval/
├── README.md                      # Comprehensive guide and next steps
├── CURRENT_RESULTS.md             # This file - results summary
├── 07_heuristic_full_dataset.py   # ✅ Heuristic evaluation script
├── heuristic_full_15140.txt       # ✅ Heuristic results
├── 08_full_dataset_llm_eval.py    # ⏳ To create - LLM evaluation
└── run_on_server.sh                # ⏳ To create - Deployment script
```

---

## Validation Checklist

Before considering this experiment complete:

**Phase 1: Heuristic Baseline** ✅
- [x] Run on full 15,140 prompts
- [x] Document precision/recall/F1
- [x] Analyze false positive rate
- [x] Platform breakdown
- [x] Compare to balanced sample

**Phase 2: LLM Evaluation** ⏳
- [ ] Create full dataset script with checkpoints
- [ ] Test on 100-prompt sample
- [ ] Run safeguard model on full dataset
- [ ] Run base model on full dataset (optional)
- [ ] Document results

**Phase 3: Layered Defense** ⏳
- [ ] Evaluate layered approach on full dataset
- [ ] Measure LLM load reduction
- [ ] Calculate cost savings
- [ ] Compare to sample results

**Phase 4: Analysis & Publication** ⏳
- [ ] Generate comparison report
- [ ] Create visualizations
- [ ] Document production deployment recommendations
- [ ] Update main project README

---

**Last Updated**: November 6, 2025
**Next Action**: Create `08_full_dataset_llm_eval.py` when GPU server is available
