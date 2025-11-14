# Jailbreak Detection Results - Model Comparison

**Date:** November 14, 2025
**Dataset:** JailbreakHub (400 samples: 200 jailbreaks, 200 benign)
**Evaluation:** Scripts 06, 07, 09, 10

---

## Three-Factor Impact Analysis: Prompting, Training, Size

```
                        RECALL (Jailbreak Detection Rate)

    100% ┤
         │
     90% ┤
         │
     80% ┤                                              ┌─────────────────┐
         │                                              │  SAFEGUARD      │
     70% ┤                   71.0%          77.0%       │  + Good Prompt  │
         │                      ●──────────────●        └─────────────────┘
     60% ┤      60.0%           │              │
         │         ●────────────┘              │
     50% ┤         │                           │
         │         │                           │
     40% ┤         │                           │
         │         │                           │
     30% ┤         │                           │
         │         │                           │
     20% ┤         │                           │
         │         │                           │
     10% ┤         │                           │
         │   ┌─────┼───────────────────────────┼────────────┐
      0% ┤   │ 1.0%│                           │            │
         │   │  ●  │  BASELINE + GOOD PROMPT   │  TBD       │
         └───┴─────┴───────────────────────────┴────────────┴─────
             │     │                           │            │
           Basic Good                         Good        Good
          Prompt Prompt                      Prompt      Prompt

                 20B Model                    120B Model

    ═══════════════════════════════════════════════════════════════
    THREE FACTORS THAT MATTER:
    ═══════════════════════════════════════════════════════════════

    1. PROMPTING STRATEGY (60x impact):
       ├─ Baseline + Basic Prompt:      1.0% recall  ❌
       └─ Baseline + SafeguardDetector: 60.0% recall ✅

    2. SPECIALIZED TRAINING (+18% relative):
       ├─ 20B Baseline + Good Prompt:   60.0% recall
       └─ 20B Safeguard + Good Prompt:  71.0% recall ✅

    3. MODEL SIZE (+8% relative):
       ├─ 20B Safeguard + Good Prompt:  71.0% recall
       └─ 120B Safeguard + Good Prompt: 77.0% recall ✅

    ═══════════════════════════════════════════════════════════════
    PRODUCTION REQUIREMENTS:
    ═══════════════════════════════════════════════════════════════

    ✅ Good Prompting (SafeguardDetector-style system prompts)
    ✅ Fine-Tuned Safeguard Models (not baseline models)
    ✅ Appropriate Model Size (20B for real-time, 120B for batch)

    Baseline + Basic Prompting = 1% recall = COMPLETELY UNSUITABLE
```

---

## Detailed Results Table

### All Evaluations Summary

| Model | Type      | Prompting           | Precision | Recall | F1    | Accuracy | Latency | Script |
|-------|-----------|---------------------|-----------|--------|-------|----------|---------|--------|
| 20B   | Baseline  | Basic (GPTOSSDetector) | 100.0%    | 1.0%   | 2.0%  | 50.5%    | 4.4s    | 09 ✅  |
| 20B   | Baseline  | SafeguardDetector   | ~81%      | 60.0%  | ~69%  | ~71%     | ~6s     | 05 ✅  |
| 20B   | Safeguard | SafeguardDetector   | 82.6%     | 71.0%  | 76.3% | 78.0%    | 1.9s    | 06 ✅  |
| 120B  | Safeguard | SafeguardDetector   | 79.0%     | 77.0%  | 78.0% | 78.2%    | 20.4s   | 07 ✅  |
| 120B  | Baseline  | Basic (GPTOSSDetector) | TBD       | TBD    | TBD   | TBD      | TBD     | 10 🟢  |

### Key Insights by Factor

**1. Prompting Impact (60x):**
- Same 20B baseline model, different prompting:
  - Basic prompting: 1% recall ❌
  - SafeguardDetector prompting: 60% recall ✅
  - **60x improvement from prompting alone**

**2. Training Impact (+18% relative):**
- Same prompting (SafeguardDetector), different training:
  - 20B Baseline: 60% recall
  - 20B Safeguard: 71% recall ✅
  - **11% absolute / 18% relative improvement**

**3. Size Impact (+8% relative):**
- Same training + prompting, different size:
  - 20B Safeguard: 71% recall
  - 120B Safeguard: 77% recall ✅
  - **6% absolute / 8% relative improvement**

---

## Key Findings

### 1. Prompting Strategy is Critical (60x Impact)
- **Same 20B model with different prompts:**
  - Basic prompting (Script 09): 1% recall ❌
  - SafeguardDetector prompting (Script 05): 60% recall ✅
- **60x improvement** from sophisticated safety-focused prompting
- Shows importance of well-designed system prompts with policy rules and examples

### 2. Specialized Training Adds Significant Value (+18% Relative)
- **Same prompting strategy, different model training:**
  - 20B Baseline + SafeguardDetector: 60% recall
  - 20B Safeguard + SafeguardDetector: 71% recall ✅
- **11% absolute improvement** (18% relative) from safety fine-tuning
- Safeguard models are optimized to work with safety-focused prompting

### 3. Model Size Provides Incremental Gains (+8% Relative)
- **Same training + prompting, different size:**
  - 20B Safeguard: 71% recall
  - 120B Safeguard: 77% recall ✅
- **6% absolute improvement** (8% relative) from increased model size
- Useful for batch processing where latency is less critical

### 4. Baseline Models Without Good Prompting Are Unsuitable
- Regular models with basic prompting are extremely risk-averse
- 1% recall = catches only 2 out of 200 jailbreaks
- Simple heuristics (69.7% F1) beat 20B baseline (2.0% F1) by 35x
- **Completely unsuitable for production safety systems**

### 5. Confusion Matrix Comparison

**20B Safeguard:**
- ✅ Caught jailbreaks: 142/200 (71%)
- ❌ Missed jailbreaks: 58/200 (29%)
- ✅ Correctly safe: 170/200 (85%)
- ⚠️ False alarms: 30/200 (15%)

**20B Baseline:**
- ❌ Caught jailbreaks: 2/200 (1%)
- ❌ Missed jailbreaks: 198/200 (99%)
- ✅ Correctly safe: 200/200 (100%)
- ⚠️ False alarms: 0/200 (0%)

**120B Safeguard:**
- ✅ Caught jailbreaks: 154/200 (77%)
- ❌ Missed jailbreaks: 46/200 (23%)
- ✅ Correctly safe: 159/200 (79.5%)
- ⚠️ False alarms: 41/200 (20.5%)

---

## Script Details

### Script 06: 20B Safeguard ✅
- **Model:** gpt-oss-safeguard:latest (20B)
- **Status:** Complete (Nov 14, 01:48 MST)
- **Runtime:** ~40 minutes
- **Log:** `experiments/jailbreak-evals/results/20b_safeguard_400_eval.log`

### Script 07: 120B Safeguard ✅
- **Model:** gpt-oss-safeguard:120b
- **Status:** Complete (Nov 13, 23:44 MST)
- **Runtime:** ~40 minutes (with 5min timeout)
- **Log:** `experiments/jailbreak-evals/results/120b_eval_07_nigel_5min.log`

### Script 09: 20B Baseline ✅
- **Model:** gpt-oss:20b (regular)
- **Status:** Complete (Nov 14, 00:08 MST)
- **Runtime:** ~1 hour 15 minutes
- **Log:** Output from local MacBook run

### Script 10: 120B Baseline 🟢
- **Model:** gpt-oss:120b (regular)
- **Status:** Running on nigel
- **Started:** Nov 14, 01:48 MST
- **ETA:** ~2 hours
- **Log:** `experiments/jailbreak-evals/results/120b_baseline_400_eval.log`

---

## Conclusion

**Production safety systems require THREE critical components:**

### 1. Sophisticated Prompting Strategy (60x impact)
- Well-designed system prompts with policy rules, definitions, and examples
- SafeguardDetector's prompting framework shows 60x improvement over basic prompts
- Single biggest factor in jailbreak detection performance

### 2. Specialized Safety Fine-Tuning (+18% relative improvement)
- Safeguard models optimized to work with safety-focused prompting
- Adds 11% absolute (18% relative) improvement on top of good prompting
- Essential for production-grade reliability

### 3. Appropriate Model Size (+8% relative improvement)
- 20B for real-time detection (71% recall, 1.9s latency)
- 120B for batch/deferred processing (77% recall, 20.4s latency)
- Size provides incremental gains but isn't sufficient alone

**Bottom Line:**
- ❌ Baseline + Basic Prompting = 1% recall (UNUSABLE)
- ⚠️ Baseline + Good Prompting = 60% recall (BETTER but not ideal)
- ✅ Safeguard + Good Prompting = 71-77% recall (PRODUCTION-READY)

**You need ALL THREE factors for production safety systems.**
