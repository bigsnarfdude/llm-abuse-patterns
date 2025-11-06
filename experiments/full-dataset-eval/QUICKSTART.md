# Quick Start Guide - Full Dataset Evaluation

**Run complete JailbreakHub evaluation when nigel.birs.ca is available**

---

## TL;DR - Run This When Nigel Is Ready

```bash
# SSH to nigel
ssh vincent@nigel.birs.ca

# Navigate to experiment
cd ~/llm-abuse-patterns/experiments/full-dataset-eval

# Run stratified evaluation (recommended, ~15 hours)
./run_on_nigel.sh stratified

# Monitor progress
tail -f logs/run_stratified_*.log
```

---

## What's Been Done ✅

1. **Heuristic Baseline** - Full 15,140 dataset evaluated (5 seconds)
   - Results: `heuristic_full_15140.txt`
   - Key finding: 26.1% recall, 2.10% FPR

2. **Sample Evaluations** - 400 balanced prompts tested
   - Base LLM: 60% recall, 86.3% precision
   - Safeguard LLM: 69% recall, 87.3% precision
   - Layered: 68% recall, 84% precision

3. **Infrastructure** - Scripts and documentation ready
   - Evaluation script with checkpoints
   - Deployment script for nigel
   - Comprehensive documentation

---

## What's Next ⏳

Run **Safeguard LLM** on full or stratified dataset to get:
- Real-world metrics (9.3% jailbreak rate)
- Accurate false positive rate
- Production-ready performance data
- Platform-specific insights

---

## Evaluation Options

### Option 1: Stratified Sample (RECOMMENDED)
**Best balance of speed and accuracy**

```bash
./run_on_nigel.sh stratified
```

- **Size**: 5,000 prompts
- **Distribution**: 9.3% jailbreaks / 90.7% benign (real-world)
- **Time**: ~15 hours
- **Why**: 12.5x larger than current sample, statistically significant

### Option 2: Full Dataset
**Most comprehensive, publication-quality**

```bash
./run_on_nigel.sh full
```

- **Size**: 15,140 prompts (complete dataset)
- **Distribution**: Real-world (9.3/90.7)
- **Time**: ~46 hours
- **Why**: Authoritative metrics, platform analysis, production-ready

### Option 3: Jailbreaks Only
**Fast recall measurement**

```bash
./run_on_nigel.sh jailbreaks
```

- **Size**: 1,405 jailbreak prompts
- **Time**: ~4.3 hours
- **Why**: Quick validation of detection capability

### Option 4: Test Run
**Verify everything works**

```bash
./run_on_nigel.sh test
```

- **Size**: 100 prompts
- **Time**: ~18 minutes
- **Why**: Sanity check before long run

---

## Step-by-Step Deployment

### 1. SSH to Nigel
```bash
ssh vincent@nigel.birs.ca
```

### 2. Sync Repository
```bash
cd ~/llm-abuse-patterns
git pull origin main
```

### 3. Check Prerequisites
```bash
# Verify Ollama running
curl http://localhost:11434/api/tags

# Check model installed
ollama list | grep gpt-oss-safeguard

# If missing, pull it
ollama pull gpt-oss-safeguard:latest
```

### 4. Navigate to Experiment
```bash
cd experiments/full-dataset-eval
```

### 5. Run Evaluation
```bash
# Test first (18 minutes)
./run_on_nigel.sh test

# If test passes, run stratified (15 hours)
./run_on_nigel.sh stratified
```

### 6. Monitor Progress
```bash
# In another SSH session
tail -f logs/run_stratified_*.log

# Check checkpoints
ls -lh checkpoints/

# View latest metrics
tail -50 logs/run_stratified_*.log
```

---

## If Something Goes Wrong

### Process Dies or Gets Interrupted

```bash
# Find latest checkpoint
ls -lt checkpoints/

# Resume from checkpoint
python3 08_full_dataset_llm_eval.py \
    --model safeguard \
    --resume checkpoints/latest.json \
    --checkpoint \
    --output logs/resumed_output.txt
```

### Ollama Not Running

```bash
# Start Ollama
ollama serve &

# Wait 5 seconds
sleep 5

# Verify
curl http://localhost:11434/api/tags
```

### Out of Memory

```bash
# Check GPU memory
nvidia-smi

# Reduce batch size
python3 08_full_dataset_llm_eval.py \
    --model safeguard \
    --sample-size 5000 \
    --stratified \
    --batch-size 50  # Smaller batches
```

### Timeout Errors

Edit `src/llm_abuse_patterns/safeguard.py` and increase timeout:
```python
# Change from 60s to 120s
timeout=120
```

---

## Expected Results

After completion, you'll have:

1. **Results file** in `logs/` directory
2. **Checkpoints** in `checkpoints/` directory
3. **Metrics** including:
   - Precision / Recall / F1
   - True/False Positives/Negatives
   - Latency statistics
   - Platform breakdown
   - Real-world impact analysis

---

## Comparing to Current Results

| Metric | Sample (400) | Stratified (5,000) | Full (15,140) |
|--------|-------------|--------------------|---------------|
| **Dataset Size** | 400 | 5,000 | 15,140 |
| **Jailbreak %** | 50% | 9.3% | 9.3% |
| **Statistical Power** | Low | Good | Best |
| **Runtime** | 40 min | ~15 hrs | ~46 hrs |
| **FP Accuracy** | Poor | Good | Best |
| **Production Ready** | No | Yes | Yes |

---

## Files You'll Generate

```
logs/
├── stratified_safeguard_20251106_143022.txt  # Full results
├── run_stratified_20251106_143022.log        # Progress log
└── ...

checkpoints/
├── batch_0001.json    # Every 100 prompts
├── batch_0002.json
├── ...
├── batch_0050.json
└── latest.json        # For resume
```

---

## Next Steps After Completion

1. **Copy results to local machine**
   ```bash
   scp vincent@nigel.birs.ca:~/llm-abuse-patterns/experiments/full-dataset-eval/logs/*.txt .
   ```

2. **Create comparison report**
   - Compare sample vs stratified vs full
   - Generate visualizations
   - Update main README

3. **Document production recommendations**
   - Optimal detection strategy
   - Cost-benefit analysis
   - Deployment configuration

4. **Publish findings**
   - GitHub repository update
   - Blog post / paper
   - Share with community

---

## Questions?

- **How long will it take?** See "Evaluation Options" above
- **Can I pause it?** Yes, Ctrl+C and resume from checkpoint
- **What if nigel reboots?** Resume from latest checkpoint
- **How much disk space?** ~2GB for full dataset with checkpoints
- **Can I run multiple models?** Yes, but sequentially (not parallel)

---

## Summary

**Recommended Path:**
1. Run `./run_on_nigel.sh test` (~18 min) ✅ Verify setup
2. Run `./run_on_nigel.sh stratified` (~15 hrs) ✅ Get real-world metrics
3. Analyze results and decide if full dataset needed
4. Optional: Run `./run_on_nigel.sh full` (~46 hrs) for publication

**Minimum Success:**
- Stratified evaluation (5,000) completed
- Results show real-world FPR and recall
- Can make production deployment decision

**Ideal Success:**
- Full dataset (15,140) completed
- Both base and safeguard models tested
- Layered defense evaluated
- Comparison report generated

---

**Ready to run when nigel is available!** 🚀
