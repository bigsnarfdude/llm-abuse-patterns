# Quick Start Guide - Full Dataset Evaluation

**Run complete JailbreakHub evaluation on GPU server with Ollama**

---

## TL;DR - Run This When GPU Server Is Ready

```bash
# SSH to your GPU server
ssh your-gpu-server

# Navigate to experiment
cd ~/llm-abuse-patterns/experiments/full-dataset-eval

# Run stratified evaluation (recommended, ~15 hours)
./run_on_server.sh stratified

# Monitor progress
tail -f logs/run_stratified_*.log
```

---

## Prerequisites

- GPU server with Ollama installed
- gpt-oss-safeguard:latest model pulled
- This repository cloned

---

## Evaluation Options

### Option 1: Stratified Sample (RECOMMENDED)
```bash
./run_on_server.sh stratified
```
- 5,000 prompts, real 9.3/90.7 distribution
- ~15 hours runtime

### Option 2: Full Dataset
```bash
./run_on_server.sh full
```
- All 15,140 prompts
- ~46 hours runtime

### Option 3: Jailbreaks Only
```bash
./run_on_server.sh jailbreaks
```
- 1,405 jailbreak prompts
- ~4.3 hours runtime

### Option 4: Test Run
```bash
./run_on_server.sh test
```
- 100 prompts
- ~18 minutes

---

## Deployment Steps

1. **SSH to server**
2. **Sync repository**: `git pull origin main`
3. **Check Ollama**: `curl http://localhost:11434/api/tags`
4. **Install model** (if needed): `ollama pull gpt-oss-safeguard:latest`
5. **Navigate**: `cd experiments/full-dataset-eval`
6. **Run**: `./run_on_server.sh stratified`

---

## Monitoring

```bash
# View progress
tail -f logs/run_stratified_*.log

# Check checkpoints
ls -lh checkpoints/
```

---

## Recovery

If interrupted, resume from checkpoint:
```bash
python3 08_full_dataset_llm_eval.py \
    --model safeguard \
    --resume checkpoints/latest.json \
    --checkpoint \
    --output logs/resumed.txt
```

---

**Ready when you are!** 🚀
