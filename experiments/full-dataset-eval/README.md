# Full Dataset Evaluation Experiment

**Comprehensive evaluation of jailbreak detection methods on the complete JailbreakHub dataset (15,140 prompts)**

## Experiment Overview

### Dataset: JailbreakHub (walledai)
- **Total Prompts**: 15,140
- **Jailbreaks**: 1,405 (9.3%)
- **Benign**: 13,735 (90.7%)
- **Source**: Reddit, Discord, websites, open-source (Dec 2022 - Dec 2023)
- **Paper**: ["Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts](https://huggingface.co/datasets/walledai/JailbreakHub)

### Detection Methods
1. **Heuristic** - Pattern matching (instant, ~0.2ms per prompt)
2. **LLM (gpt-oss:20b)** - Base reasoning model (~13s per prompt)
3. **LLM (gpt-oss-safeguard:latest)** - Official OpenAI safeguard model (~11s per prompt)
4. **Layered Defense** - Heuristic → LLM (adaptive latency)

---

## Current Status

### ✅ Completed: Heuristic Full Dataset Evaluation

**Script**: `07_heuristic_full_dataset.py`
**Results**: `heuristic_full_15140.txt`
**Runtime**: 5.35 seconds
**Date**: November 6, 2025

#### Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| **Precision** | 55.95% | TP / (TP + FP) |
| **Recall** | 26.12% | TP / (TP + FN) - Misses 73.9% of jailbreaks |
| **F1 Score** | 35.61% | Harmonic mean |
| **Accuracy** | 91.24% | High due to 90.7% benign baseline |
| **False Positive Rate** | 2.10% | 289/13,735 benign flagged |
| **Throughput** | 2,832 prompts/sec | Extremely fast |

#### Confusion Matrix
```
True Positives:     367  (caught jailbreaks)
False Negatives:  1,038  (missed jailbreaks) ⚠️
True Negatives:  13,446  (correctly safe)
False Positives:   289  (false alarms)
```

#### Real-World Impact
- **Detection Rate**: 26.1% (catches only 1 in 4 jailbreaks)
- **False Alarms**: 210 per 10,000 benign prompts
- **Best for**: Fast first-layer filtering, blocking obvious attacks

#### Platform Breakdown
| Platform | Prompts | Precision | Recall | Notes |
|----------|---------|-----------|--------|-------|
| Reddit | 1,055 | 94.2% | 30.6% | Best precision, good recall |
| Discord | 1,106 | 84.8% | 26.2% | Balanced performance |
| Website | 12,763 | 30.6% | 22.4% | Poor precision (many FPs) |
| Open Source | 216 | 0.0% | 0.0% | No jailbreaks detected |

---

## 🚀 Next Steps: Full LLM Evaluation

### Comparison: Sample vs Full Dataset

| Evaluation | Heuristic | Base LLM | Safeguard LLM | Layered |
|------------|-----------|----------|---------------|---------|
| **400-sample (balanced 50/50)** | ✅ Done | ✅ Done | ✅ Done | ✅ Done |
| **15,140-full (real 9.3/90.7)** | ✅ Done | ⏳ Pending | ⏳ Pending | ⏳ Pending |

### Why Full Dataset Matters

**Current Sample (400 prompts, 50/50 balanced):**
- Artificial 50% jailbreak rate
- False positive rate is understated
- Doesn't reflect real-world production traffic

**Full Dataset (15,140 prompts, 9.3/90.7 real distribution):**
- ✅ Real-world 9.3% jailbreak prevalence
- ✅ Accurate false positive impact (90.7% benign traffic)
- ✅ Statistical significance (37x larger sample)
- ✅ Platform-specific performance insights
- ✅ Production-realistic metrics

### Time Estimates

| Method | Full Dataset (15,140) | Notes |
|--------|----------------------|-------|
| Heuristic | ~5 seconds | ✅ **COMPLETE** |
| Base LLM (gpt-oss:20b) | ~55 hours | 13s/prompt × 15,140 |
| Safeguard LLM | ~46 hours | 11s/prompt × 15,140 |
| Layered Defense | ~40 hours | 12.5% filtered by heuristic |

### Optimization Strategies

#### Option 1: Full Dataset Evaluation (Overnight on Server)
```bash
# Run on nigel.birs.ca (GPU server)
nohup python 08_full_dataset_llm_eval.py --model safeguard > full_eval.log 2>&1 &

# Monitor progress
tail -f full_eval.log
```

**Benefits:**
- Complete, authoritative metrics
- Real-world distribution preserved
- Platform-specific insights
- Publication-ready results

**Requirements:**
- nigel.birs.ca server running
- Ollama with gpt-oss-safeguard:latest
- ~48 hours uninterrupted runtime
- ~200GB available storage

#### Option 2: Stratified Sampling (Faster, Statistically Valid)
```bash
# 5,000 prompts preserving 9.3%/90.7% ratio (~15 hours)
python 08_full_dataset_llm_eval.py --model safeguard --sample-size 5000 --stratified
```

**Benefits:**
- Preserves real-world distribution
- 12.5x larger than current sample
- Manageable runtime (~15 hours)
- Statistically significant

#### Option 3: Jailbreaks-Only Evaluation (Focus on Recall)
```bash
# Test only 1,405 jailbreaks (~4.3 hours)
python 08_full_dataset_llm_eval.py --model safeguard --jailbreaks-only
```

**Benefits:**
- Measures recall (false negative rate)
- Quick evaluation of detection capability
- Identifies missed attack patterns
- Useful for model comparison

#### Option 4: Incremental Batch Evaluation (Resumable)
```bash
# Process in batches, save checkpoints
python 08_full_dataset_llm_eval.py --model safeguard --batch-size 1000 --checkpoint
```

**Benefits:**
- Resume after interruption
- Progress tracking
- Risk mitigation
- Flexible scheduling

---

## Deployment: Running on Nigel Server

### Prerequisites
```bash
# SSH to nigel
ssh vincent@nigel.birs.ca

# Check Ollama installation
ollama list

# Pull safeguard model if needed
ollama pull gpt-oss-safeguard:latest

# Verify GPU availability
nvidia-smi
```

### Setup
```bash
# Clone or sync repository
cd ~/llm-abuse-patterns
git pull origin main

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install datasets  # For HuggingFace datasets

# Navigate to experiment directory
cd experiments/full-dataset-eval
```

### Execution
```bash
# Option 1: Full dataset (recommended for overnight run)
nohup python 08_full_dataset_llm_eval.py \
    --model safeguard \
    --output full_safeguard_15140.txt \
    > run.log 2>&1 &

# Save process ID
echo $! > pid.txt

# Option 2: Stratified sample (faster alternative)
nohup python 08_full_dataset_llm_eval.py \
    --model safeguard \
    --sample-size 5000 \
    --stratified \
    --output stratified_5000.txt \
    > run.log 2>&1 &

# Option 3: Incremental with checkpoints (safest)
nohup python 08_full_dataset_llm_eval.py \
    --model safeguard \
    --batch-size 1000 \
    --checkpoint \
    --checkpoint-dir checkpoints/ \
    --output incremental_safeguard.txt \
    > run.log 2>&1 &
```

### Monitoring
```bash
# View live progress
tail -f run.log

# Check process status
ps -p $(cat pid.txt)

# Estimate completion time
grep "Progress:" run.log | tail -5

# Check partial results (if checkpoints enabled)
ls -lh checkpoints/
```

### Recovery
```bash
# If process dies, resume from checkpoint
python 08_full_dataset_llm_eval.py \
    --model safeguard \
    --resume checkpoints/latest.json \
    --output incremental_safeguard.txt
```

---

## Expected Outputs

### File Structure After Completion
```
experiments/full-dataset-eval/
├── README.md                           # This file
├── 07_heuristic_full_dataset.py        # ✅ Heuristic script
├── 08_full_dataset_llm_eval.py         # ⏳ LLM evaluation script (to create)
├── heuristic_full_15140.txt            # ✅ Heuristic results
├── base_llm_full_15140.txt             # ⏳ Base LLM results
├── safeguard_llm_full_15140.txt        # ⏳ Safeguard LLM results
├── layered_full_15140.txt              # ⏳ Layered defense results
├── comparison_report.md                # ⏳ Comparative analysis
├── checkpoints/                        # ⏳ Incremental saves (optional)
│   ├── batch_0001.json
│   ├── batch_0002.json
│   └── ...
└── visualizations/                     # ⏳ Plots and charts (optional)
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── platform_comparison.png
```

### Results Format
Each evaluation will produce:
1. **Detailed metrics** (precision, recall, F1, accuracy, FPR)
2. **Confusion matrix** (TP, FP, TN, FN)
3. **Latency statistics** (mean, median, p95, p99)
4. **Platform breakdown** (Reddit, Discord, websites, open-source)
5. **Real-world impact analysis** (detection rate, false alarm rate)
6. **Layered efficiency** (heuristic filtering, LLM load reduction)

---

## Research Questions to Answer

### Detection Performance
1. **How does recall improve with full dataset?**
   - Current sample: 60-69% recall
   - Expected full: Similar, but with confidence intervals

2. **What is the true false positive rate at 90.7% benign traffic?**
   - Current heuristic: 2.10% FPR (289/13,735)
   - Expected LLM: <1% FPR based on sample results

3. **Which platforms have highest jailbreak concentration?**
   - Discord vs Reddit vs websites
   - Platform-specific detection strategies

### Layered Defense Efficiency
4. **What % of traffic requires expensive LLM inference?**
   - Current: 87.5% (heuristic filters 12.5%)
   - Potential optimization: Tune heuristic threshold

5. **What's the cost-accuracy tradeoff?**
   - Heuristic: Free, 26% recall
   - LLM: Expensive, 60-69% recall
   - Layered: Balanced, ~68% recall with 12.5% instant filtering

### Production Deployment Insights
6. **At 10,000 req/day, how many false alarms?**
   - 90.7% benign = 9,070 benign/day
   - 2.10% heuristic FPR = 190 false alarms/day
   - <1% LLM FPR = <91 false alarms/day

7. **What's the latency distribution in production?**
   - P50, P95, P99 latencies
   - Tail latency for timeout configuration

---

## Success Metrics

### Minimum Viable Results
- ✅ Heuristic full dataset completed
- ⏳ At least one LLM method on full dataset OR
- ⏳ Stratified sample (5,000+) with both LLM methods

### Ideal Complete Results
- ✅ Heuristic full dataset
- ⏳ Base LLM (gpt-oss:20b) full dataset
- ⏳ Safeguard LLM (gpt-oss-safeguard) full dataset
- ⏳ Layered defense full dataset
- ⏳ Comparative analysis report
- ⏳ Production deployment recommendations

---

## Timeline

### Phase 1: Preparation (Completed ✅)
- [x] Create experiment directory
- [x] Document current results
- [x] Write execution plan
- [x] Create heuristic baseline

### Phase 2: Server Setup (When Nigel Available)
- [ ] SSH to nigel.birs.ca
- [ ] Verify Ollama + GPU setup
- [ ] Pull gpt-oss-safeguard:latest model
- [ ] Clone/sync repository
- [ ] Install dependencies

### Phase 3: Execution (Estimated 48-72 hours)
- [ ] Create full dataset LLM evaluation script
- [ ] Test on small sample (100 prompts)
- [ ] Launch full evaluation (overnight)
- [ ] Monitor progress daily
- [ ] Backup intermediate results

### Phase 4: Analysis (After Completion)
- [ ] Generate comparative report
- [ ] Create visualizations
- [ ] Document findings
- [ ] Update main README
- [ ] Publish results

---

## Notes

### Why This Matters
**Current sample (400)** shows what's possible.
**Full dataset (15,140)** shows what's real.

The difference between 50/50 balanced and 9.3/90.7 real distribution is critical:
- False positive rate has 9x more impact
- Accuracy metrics change dramatically
- Production cost estimates are accurate
- Platform-specific insights emerge

### Hardware Requirements
**Minimum:**
- GPU with 16GB VRAM (RTX 4070 Ti or better)
- 32GB system RAM
- 200GB free disk space
- Stable network connection

**Optimal:**
- GPU with 24GB+ VRAM (RTX 4090, A5000)
- 64GB system RAM
- NVMe SSD storage
- Dedicated server (nigel.birs.ca)

### Cost-Benefit Analysis
**Without full evaluation:**
- Uncertainty in production deployment
- Risk of unexpected false positive rate
- Unknown platform-specific behavior
- Limited confidence in metrics

**With full evaluation:**
- Production-ready metrics
- Accurate cost projections
- Platform optimization opportunities
- Publishable research results

---

## Contact

For questions about this experiment:
- **Repository**: llm-abuse-patterns
- **Experiment**: experiments/full-dataset-eval/
- **Status**: Phase 1 complete, Phase 2 pending nigel availability

---

## References

1. **JailbreakHub Dataset**: https://huggingface.co/datasets/walledai/JailbreakHub
2. **Paper**: "Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts
3. **GPT-OSS Safeguard**: https://github.com/openai/gpt-oss-safeguard
4. **Ollama**: https://ollama.com/library/gpt-oss-safeguard

---

**Last Updated**: November 6, 2025
**Status**: Ready for nigel server deployment
