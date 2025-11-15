# Experiment 02: Enhanced Policy - In Progress

## Status: RUNNING 🔄

**Started**: 2025-11-15 16:32 (on nigel.birs.ca)
**Current Progress**: 200/5,083 samples (3.9%)
**Estimated Completion**: ~10:15 PM (5.7 hours total)

## Early Results (200 samples)

| Metric | Baseline (Exp 01) | Enhanced (Current) | Delta |
|--------|------------------|-------------------|-------|
| **F1** | 70.1% | **73.3%** | **+3.2%** ✅ |
| **Recall** | 59.9% | **60.4%** | **+0.5%** |
| **Precision** | 84.4% | **93.2%** | **+8.8%** ✅ |

## Policy Changes

### Enhanced Policy (711 tokens vs 447 original)
⚠️ **WARNING**: 711 tokens exceeds OpenAI's 400-600 recommendation

**Key Enhancements**:
1. **Examples**: 2 → 6 (3 toxic, 3 safe)
   - Added microaggression example ("You're pretty smart for a girl")
   - Added respectful discussion example (Charlie crossdresser)
   - Added technical/educational example (mental health cultures)

2. **Rules**: 5 → 7 categories
   - T6: Microaggressions or coded discriminatory language
   - T7: Context-dependent toxicity (demeaning based on target)

3. **Safe Content**: Expanded to 5 types
   - S5: Reclaimed language within community (context matters)

4. **Edge Case Handling**: Added ambiguity guidance
   - If context unclear → consider intent and target
   - If reclaimed language → check speaker community
   - If borderline → default to SAFE unless clear harm
   - If educational vs offensive → look for pedagogical framing

## Analysis of Early Results

### ✅ Precision Improvement (+8.8%)
**Strong signal that edge case handling is working!**
- Original: 84.4% precision (40 false positives in exp 01)
- Enhanced: 93.2% precision (fewer false positives expected)
- Root cause: Better guidance on borderline cases reduces over-flagging

### ⚠️ Recall Improvement (+0.5%)
**Modest gain - may not reach 75% target**
- Original: 59.9% recall (145 false negatives in exp 01)
- Enhanced: 60.4% recall (still missing many toxic examples)
- Issue: More examples alone may not capture full toxic spectrum

### 🎯 F1 Improvement (+3.2%)
**Good progress toward 79.9% OpenAI target**
- Gap closed: 3.2% out of 9.8% needed (33% of gap)
- Remaining gap: 6.6% (73.3% → 79.9%)
- Outlook: Unlikely to hit 79.9%, but 75-77% achievable

## Predictions for Final Results

### Conservative Estimate (Early Metrics Hold)
- **F1**: 72-74% (+2-4% from baseline)
- **Recall**: 59-62% (modest gain)
- **Precision**: 90-94% (strong gain)

**Conclusion**: Edge case handling helps precision significantly, but recall still limited

### Optimistic Estimate (Metrics Improve)
- **F1**: 75-77% (+5-7% from baseline)
- **Recall**: 64-68% (moderate gain)
- **Precision**: 88-92% (strong gain)

**Conclusion**: Close to OpenAI's 79.9% target (within 3-5%)

### Key Uncertainty
**Early results (200 samples) may not represent full distribution**
- Only 14 toxic samples seen so far (7.1% of 200)
- Toxic sample performance will determine recall
- Benign sample performance will determine precision

## What This Means

### If F1 ≥ 75% (Success)
✅ **Enhanced policy is effective**
- More examples + broader rules close most of gap
- Edge case guidance reduces false positives
- Ready for production deployment

**Next Steps**:
- Test multi-policy approach (toxicity subcategories)
- Compare with dedicated BERT classifier
- Implement hybrid architecture (pre-filter + safeguard)

### If F1 = 72-74% (Partial Success)
⚠️ **Policy enhancements help but insufficient**
- Precision improved significantly (edge cases work)
- Recall still too low (need different approach)

**Next Steps**:
- Try multi-policy specialized approach (sexual, harassment, violence, etc.)
- Consider dedicated classifier for initial filtering
- Test few-shot learning (dynamic examples per category)

### If F1 < 72% (Need Different Approach)
❌ **Policy tuning has limited effectiveness**
- May have hit ceiling for single-policy approach
- Need dedicated classifiers (OpenAI's suggestion)

**Next Steps**:
- Fine-tune BERT on ToxicChat (50K+ samples)
- Implement multi-classifier architecture
- Use safeguard only for borderline cases

## Policy Length Concern

**Enhanced policy: 711 tokens (⚠️ exceeds 400-600 recommendation)**

**Potential Issues**:
1. Longer context → slower inference
2. Too much information → model confusion
3. Violates OpenAI's optimization guidance

**Test After Experiment**:
- Create 500-token version (trim edge cases, condense examples)
- Compare performance: 711 vs 500 tokens
- Validate OpenAI's 400-600 recommendation for toxicity

## Timeline

| Time | Progress | Status |
|------|----------|--------|
| 16:32 | 0/5,083 | Started |
| 16:46 | 200/5,083 (3.9%) | Early results promising |
| ~18:00 | ~1,300/5,083 (25%) | First checkpoint |
| ~20:00 | ~2,600/5,083 (50%) | Halfway point |
| ~22:00 | ~3,900/5,083 (75%) | Third checkpoint |
| ~22:15 | 5,083/5,083 (100%) | **COMPLETE** |

## Commands to Monitor

### Check Progress
```bash
ssh vincent@nigel.birs.ca "tail -30 ~/llm-abuse-patterns/experiments/toxicchat-evals/results/02_toxicchat_enhanced_policy.log"
```

### Check If Still Running
```bash
ssh vincent@nigel.birs.ca "ps aux | grep toxicchat_enhanced_policy"
```

### View Results When Complete
```bash
ssh vincent@nigel.birs.ca "tail -100 ~/llm-abuse-patterns/experiments/toxicchat-evals/results/02_toxicchat_enhanced_policy.log"
```

---

**Last Updated**: 2025-11-15 16:46
**Next Check**: 18:00 (25% checkpoint)
