# Jailbreak Benchmark Integration Plan

## Goal
Integrate real academic benchmarks to properly evaluate layered defenses

## Datasets to Integrate

### 1. JailbreakBench (Priority 1 - Best for us)
**Why:** 
- Curated 100 behaviors
- Standardized evaluation
- Used in papers
- Manageable size for M2 Mac

**Integration:**
```python
# Install
pip install jailbreakbench

# Use
import jailbreakbench as jbb
dataset = jbb.read_dataset()
# Gives us: prompts, ground truth labels, categories
```

### 2. HarmBench (Priority 2)
**Why:**
- 400+ behaviors
- Red-teaming focus
- Good for testing robustness

**Integration:**
```python
from datasets import load_dataset
dataset = load_dataset("centerforaisafety/harmbench")
```

### 3. JailbreakDB (Optional - Too large)
**Why Not First:**
- 445K prompts - too big for M2 testing
- Could use subset for stress testing

## Proposed Evaluation Script

```python
# 04_benchmark_evaluation.py

class BenchmarkEvaluator:
    def __init__(self):
        self.heuristic = HeuristicDetector()
        self.llm = RealLLMDetector()
    
    def load_jailbreakbench(self):
        """Load JailbreakBench dataset"""
        import jailbreakbench as jbb
        return jbb.read_dataset()
    
    def evaluate_single(self, prompt):
        """Test single prompt through layered defense"""
        # Layer 1: Heuristic
        heuristic_result = self.heuristic.detect(prompt)
        if heuristic_result.is_jailbreak:
            return {
                "blocked_by": "heuristic",
                "latency": heuristic_result.latency_ms,
                "is_jailbreak": True
            }
        
        # Layer 2: Real LLM (only if heuristic passed)
        llm_result = self.llm.detect(prompt)
        return {
            "blocked_by": "llm" if llm_result.is_jailbreak else "none",
            "latency": heuristic_result.latency_ms + llm_result.latency_ms,
            "is_jailbreak": llm_result.is_jailbreak
        }
    
    def evaluate_layered(self, dataset):
        """Evaluate layered defense on benchmark"""
        results = {
            "heuristic_only": {},
            "llm_only": {},
            "layered": {}
        }
        
        for prompt, ground_truth in dataset:
            # Test each layer independently
            h_result = self.heuristic.detect(prompt)
            l_result = self.llm.detect(prompt)
            
            # Test layered (sequential)
            layered_result = self.evaluate_single(prompt)
            
            # Store results
            # ... calculate metrics
        
        return results
```

## Metrics to Track

### Per-Layer Metrics:
- True Positive Rate (Recall)
- False Positive Rate
- Precision
- F1 Score
- Latency (p50, p95, p99)

### Layered Metrics:
- **Combined Recall:** How many jailbreaks caught by either layer
- **Heuristic Efficiency:** % blocked by fast layer
- **LLM Load:** % that needed slow layer
- **Average Latency:** Weighted by which layer caught it
- **Cost Efficiency:** Assuming LLM is expensive

### Example Output:
```
JAILBREAKBENCH EVALUATION (100 behaviors)

Layer 1 (Heuristic):
  Caught: 35/100 jailbreaks (35%)
  Latency: 0.5ms avg
  False Positives: 2/50 benign (4%)

Layer 2 (Real-LLM):
  Caught: 88/100 jailbreaks (88%)
  Latency: 6.2s avg
  False Positives: 0/50 benign (0%)

Layered Defense:
  Total Caught: 92/100 jailbreaks (92%)  ← New unique catches
  Avg Latency: 2.1s (35% @ 0.5ms, 57% @ 6.2s)
  False Positives: 2/50 benign (4%)
  
Efficiency:
  - 35% blocked instantly (<1ms)
  - 57% caught by LLM (6.2s)
  - 8% escaped both layers
  - LLM only processes 65% of traffic
```

## Implementation Steps

1. ✅ Install JailbreakBench
2. ✅ Create benchmark loader
3. ✅ Modify evaluation to track per-prompt results
4. ✅ Implement layered evaluation logic
5. ✅ Calculate combined metrics
6. ✅ Generate comparison report
7. ✅ Test on M2 Mac (expect ~10-15 minutes for 100 prompts)

## Questions for You

1. Start with JailbreakBench (100 prompts)?
2. Also integrate HarmBench (400 prompts)?
3. Want to see per-category breakdown (DAN, nested, obfuscation, etc.)?
4. Should we track cost (if LLM had API cost)?

