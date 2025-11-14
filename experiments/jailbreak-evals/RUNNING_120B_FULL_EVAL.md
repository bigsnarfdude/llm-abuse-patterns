# 120B Full Dataset Evaluation - IN PROGRESS

**Started:** November 12, 2025 @ 13:39 (1:39 PM MST)
**Server:** nigel.birs.ca
**Screen Session:** `120b-full-eval`
**Estimated Duration:** 30-40 hours
**Expected Completion:** November 14, 2025 (Thursday evening)

## Configuration

- **Model:** gpt-oss-safeguard:120b (65GB)
- **Reasoning Effort:** MEDIUM (baseline performance)
- **Dataset:** JailbreakHub - 4,355 prompts
  - Jailbreaks: 1,405
  - Benign: 2,950
- **Script:** experiments/jailbreak-evals/08_jailbreakhub_full_120b_eval.py
- **Log File:** experiments/jailbreak-evals/results/120b_full_evaluation_results.log

## Purpose

This is a comprehensive baseline evaluation of the 120B model on the full balanced JailbreakHub dataset, using **MEDIUM reasoning effort** (normal performance).

This differs from the previous 400-prompt evaluation which used **HIGH reasoning effort** for Layer 3 validation.

## Monitoring Commands

### Check Progress
```bash
ssh vincent@nigel.birs.ca "tail -20 ~/llm-abuse-patterns/experiments/jailbreak-evals/results/120b_full_evaluation_results.log"
```

### Check Screen Session
```bash
ssh vincent@nigel.birs.ca "screen -ls"
```

### Attach to Screen (to watch live)
```bash
ssh vincent@nigel.birs.ca
screen -r 120b-full-eval
# Press Ctrl+A then D to detach without stopping
```

### Count Progress
```bash
ssh vincent@nigel.birs.ca "grep -c 'Progress:' ~/llm-abuse-patterns/experiments/jailbreak-evals/results/120b_full_evaluation_results.log"
```

### Estimate Time Remaining
The evaluation runs in three phases:
1. **Heuristic:** 4,355 prompts (~6 seconds total) ✅ FAST
2. **Real-LLM (120B):** 4,355 prompts (~22.6 hours) ⏳ IN PROGRESS NEXT
3. **Layered:** 4,355 prompts (~12-15 hours depending on heuristic filtering)

## Timeline

| Phase | Prompts | Est. Time | Status |
|-------|---------|-----------|--------|
| Heuristic | 4,355 | ~6 sec | ✅ Running (3950/4355) |
| Real-LLM 120B | 4,355 | ~22.6 hrs | ⏳ Queued |
| Layered Defense | 4,355 | ~12-15 hrs | ⏳ Queued |
| **Total** | **13,065 tests** | **~30-40 hrs** | **🟢 ACTIVE** |

## Expected Results

Based on the 400-prompt evaluation, we expect:

### 120B Model (MEDIUM reasoning)
- Recall: ~75-80%
- Precision: ~83-87%
- F1 Score: ~79-83%
- Median Latency: ~18-20s per prompt

This will provide the comprehensive baseline for 120B performance on the full dataset.

## Comparison to Previous Runs

| Evaluation | Prompts | Model | Reasoning | Recall | F1 | Status |
|------------|---------|-------|-----------|--------|----|-|
| Script 07 | 400 | 120B | HIGH | 79% | 82.1% | ✅ Complete |
| **Script 08** | **4,355** | **120B** | **MEDIUM** | **TBD** | **TBD** | **🟢 RUNNING** |

## Notes

- The script correctly labels all output as "120B"
- Verification document created for the 400-prompt run
- This run uses MEDIUM reasoning for baseline (not HIGH for Layer 3)
- Results will be saved to: `120b_full_evaluation_results.log`

---

**Monitor Status:** Check progress with the commands above
**ETA:** Thursday, November 14, 2025 evening
