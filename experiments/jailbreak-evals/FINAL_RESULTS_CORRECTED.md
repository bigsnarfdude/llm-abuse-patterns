# CORRECTED RESULTS: Baseline vs Safeguard Models

**Date:** November 14, 2025
**Critical Discovery:** Original scripts had parsing bug - baseline models weren't being evaluated correctly

---

## Summary: What Changed

### Before Fix (Parsing Bug)
Scripts only read `content` field, which is empty for baseline models:
- **20B Baseline:** 1% recall (2/200 caught)
- **120B Baseline:** 0% recall (0/200 caught)
- **Conclusion:** "Baseline models are useless for safety"

### After Fix (Correct Parsing)
Scripts read both `content` AND `thinking` fields:
- **20B Baseline:** 74.0% recall (148/200 caught) ✅
- **120B Baseline:** Running on nigel (~2 hours)
- **Conclusion:** "Baseline models ARE capable of safety detection!"

---

## Corrected Performance Comparison

| Model | Recall | Precision | F1 Score | Parsing Method |
|-------|--------|-----------|----------|----------------|
| **20B Baseline (broken parsing)** | 1.0% | 100.0% | 2.0% | Content-only ❌ |
| **20B Baseline (fixed parsing)** | 74.0% | 79.6% | 76.7% | Content + Thinking ✅ |
| **20B Safeguard** | 71.0% | 82.6% | 76.3% | Content (proper format) ✅ |
| **120B Safeguard** | 77.0% | 79.0% | 78.0% | Content (proper format) ✅ |

---

## Key Findings (REVISED)

### 1. Baseline Models ARE Capable
**Previous belief:** Baseline models can't detect jailbreaks (1% recall)
**Reality:** Baseline models CAN detect jailbreaks (74% recall)
**Issue:** They don't format output properly - answer is in thinking field, not content

### 2. Baseline vs Safeguard (Apples-to-Apples)
When both parsed correctly:
- 20B Baseline: 74.0% recall
- 20B Safeguard: 71.0% recall
- **Baseline is 3% BETTER!**

### 3. Safeguard Model Advantage
Safeguard fine-tuning provides:
- ✅ **Proper output formatting** (answer in content field)
- ✅ **No thinking fallback needed** (production-ready)
- ✅ **Consistent response structure** (easier to parse)
- ⚠️ **Slightly lower recall** (-3%) but more reliable

### 4. Production Implications

**Option A: Baseline + Thinking Fallback**
- Pros: 74% recall (highest), already have the model
- Cons: Requires complex parsing, less reliable, not optimized for task

**Option B: Safeguard Model**
- Pros: 71% recall, proper formatting, production-ready, task-optimized
- Cons: 3% lower recall, requires specific model

**Recommendation:** Use safeguard model for production
- Proper formatting > 3% recall difference
- Easier integration and maintenance
- Purpose-built for safety tasks

---

## Confusion Matrix Comparison

### 20B Baseline (Fixed Parsing)
```
True Positives:  148/200 (74.0% recall)
False Negatives: 52/200  (26.0% missed)
True Negatives:  162/200 (81.0% specificity)
False Positives: 38/200  (19.0% false alarm rate)
```

### 20B Safeguard
```
True Positives:  142/200 (71.0% recall)
False Negatives: 58/200  (29.0% missed)
True Negatives:  170/200 (85.0% specificity)
False Positives: 30/200  (15.0% false alarm rate)
```

**Analysis:**
- Baseline catches 6 MORE jailbreaks (148 vs 142)
- Safeguard has 8 FEWER false alarms (30 vs 38)
- Baseline: higher recall, more false positives
- Safeguard: balanced precision/recall

---

## What We Learned

### The Bug
Both baseline and safeguard models use Harmony format (thinking + content):
- **Safeguard models:** Put answer in `content` field ✅
- **Baseline models:** Put answer in `thinking` field, leave `content` empty ❌
- **Original scripts:** Only read `content` → missed baseline answers

### The Fix
Updated parsing logic:
1. Try to read `content` first (works for safeguard)
2. Fallback to `thinking` if content empty (works for baseline)
3. Look for explicit conclusions: "so answer JAILBREAK", "output SAFE"

### The Insight
Safeguard fine-tuning teaches:
1. ~~Better safety reasoning~~ (both can reason well)
2. ✅ **Proper output formatting** (answer in content field)
3. ✅ **Task optimization** (consistent, production-ready responses)

The value isn't in capability - it's in reliability and formatting!

---

## Updated Recommendations

### For Research/Experimentation
- **Use baseline models** with thinking fallback parsing
- Get 74% recall with models you already have
- Good for understanding model capabilities

### For Production Systems
- **Use safeguard models** (gpt-oss-safeguard:latest or :120b)
- Accept 3% lower recall for proper formatting
- Easier integration, more reliable, purpose-built
- No complex thinking field parsing needed

### Three-Tier Architecture (Validated)
```
Layer 1: Heuristic (0.3ms)           → 43% blocked instantly
Layer 2: 20B Safeguard (1.9s)        → 71% recall (real-time)
Layer 3: 120B Safeguard (20.4s)      → 77% recall (batch/deferred)
```

All layers use safeguard models for consistent formatting and reliability.

---

## Files

**Results:**
- `experiments/jailbreak-evals/results/09_baseline_fixed_400_eval.log` - 20B baseline (74% recall)
- `experiments/jailbreak-evals/results/120b_baseline_fixed_400_eval.log` - 120B baseline (running)

**Scripts:**
- `09_jailbreakhub_gptoss_20b_baseline.py` - Fixed with thinking fallback
- `10_jailbreakhub_gptoss_120b_baseline.py` - Fixed with thinking fallback

**Documentation:**
- `THINKING_MODEL_FIX.md` - Technical analysis of the bug
- `FINAL_RESULTS_CORRECTED.md` - This file

---

## Conclusion

The 1% → 74% improvement wasn't from better prompting or model capability - it was **fixing a parsing bug**.

**Key Takeaways:**
1. Baseline models CAN detect jailbreaks (74% recall with correct parsing)
2. Safeguard models offer proper formatting, not better reasoning
3. For production, use safeguard models despite 3% lower recall
4. Proper output formatting > slight recall improvements

**The real value of safeguard fine-tuning:** Making models production-ready with consistent, properly-formatted responses.
