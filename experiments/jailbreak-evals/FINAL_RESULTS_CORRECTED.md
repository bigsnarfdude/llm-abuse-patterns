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
| **120B Baseline (fixed parsing)** | **86.0%** | 74.8% | **80.0%** | Content + Thinking ✅ |
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
- **20B Baseline: 74.0% recall vs 20B Safeguard: 71.0% recall** → Baseline is 3% BETTER!
- **120B Baseline: 86.0% recall vs 120B Safeguard: 77.0% recall** → Baseline is 9% BETTER!

**SHOCKING FINDING:** Baseline models consistently OUTPERFORM safeguard models in recall when parsed correctly!

### 3. Safeguard Model Advantage
Safeguard fine-tuning provides:
- ✅ **Proper output formatting** (answer in content field)
- ✅ **No thinking fallback needed** (production-ready)
- ✅ **Consistent response structure** (easier to parse)
- ⚠️ **Slightly lower recall** (-3%) but more reliable

### 4. Production Implications

**Option A: Baseline + Thinking Fallback**
- Pros: **86% recall (120B) - HIGHEST!**, already have the model, better detection
- Cons: Requires thinking field parsing, more complex integration

**Option B: Safeguard Model**
- Pros: 77% recall (120B), proper formatting, production-ready, consistent responses
- Cons: **9% lower recall**, requires specific model, misses more jailbreaks

**Revised Recommendation:**
- **Research/High-Security**: Use baseline 120B (86% recall) with thinking parsing - Best detection!
- **Production/Ease-of-Use**: Use safeguard 120B (77% recall) - Accept 9% lower recall for simpler integration

**Critical Trade-off:** 9% more jailbreaks caught (baseline) vs easier integration (safeguard)

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

### 120B Baseline (Fixed Parsing)
```
True Positives:  172/200 (86.0% recall) 🔥 BEST!
False Negatives: 28/200  (14.0% missed)
True Negatives:  142/200 (71.0% specificity)
False Positives: 58/200  (29.0% false alarm rate)
```

### 120B Safeguard
```
True Positives:  154/200 (77.0% recall)
False Negatives: 46/200  (23.0% missed)
True Negatives:  158/200 (79.0% specificity)
False Positives: 42/200  (21.0% false alarm rate)
```

**Analysis:**
- **120B Baseline catches 18 MORE jailbreaks than 120B Safeguard** (172 vs 154) - 9% better!
- **20B Baseline catches 6 MORE jailbreaks than 20B Safeguard** (148 vs 142) - 3% better!
- Safeguard models have better precision (fewer false positives)
- Baseline models have superior recall (catch more real jailbreaks)
- **Trade-off**: Baseline = best detection, Safeguard = fewer false alarms

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
- `experiments/jailbreak-evals/results/120b_baseline_fixed_400_eval.log` - 120B baseline (86% recall)

**Scripts:**
- `09_jailbreakhub_gptoss_20b_baseline.py` - Fixed with thinking fallback
- `10_jailbreakhub_gptoss_120b_baseline.py` - Fixed with thinking fallback

**Documentation:**
- `THINKING_MODEL_FIX.md` - Technical analysis of the bug
- `FINAL_RESULTS_CORRECTED.md` - This file

---

## Conclusion

The 0-1% → 74-86% improvement wasn't from better prompting or model capability - it was **fixing a parsing bug**.

**Key Takeaways:**
1. **Baseline models OUTPERFORM safeguard models in recall!**
   - 20B: 74% vs 71% (3% better)
   - 120B: **86% vs 77% (9% better!)**
2. Safeguard models offer proper formatting and better precision, not better recall
3. **For maximum security:** Use baseline 120B (86% recall) with thinking field parsing
4. **For production ease:** Use safeguard 120B (77% recall) - simpler integration
5. **Critical trade-off:** 9% more jailbreaks caught vs easier integration

**The real value of safeguard fine-tuning:** Making models production-ready with consistent, properly-formatted responses - NOT better detection capability.

**Shocking discovery:** Baseline models are BETTER at jailbreak detection than fine-tuned safeguard models!
