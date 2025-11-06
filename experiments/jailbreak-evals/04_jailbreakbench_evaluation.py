#!/usr/bin/env python3
"""
JailbreakBench Evaluation - Academic Baseline
----------------------------------------------
Evaluate detection methods on JailbreakBench (NeurIPS 2024 benchmark)
Tests heuristic, real LLM, and LAYERED defense on 100 harmful behaviors.

This provides REAL academic baseline numbers, not toy examples.

Requirements:
- pip install jailbreakbench
- Ollama running with gpt-oss:20b model
"""

import sys
sys.path.insert(0, 'src')

import time
import re
from typing import List, Dict
from dataclasses import dataclass
import statistics
from collections import defaultdict

# Import JailbreakBench
import jailbreakbench as jbb

# Import our real detectors
from llm_abuse_patterns.safeguard import SafeguardDetector


@dataclass
class DetectionResult:
    """Result from a detection method"""
    is_jailbreak: bool
    confidence: float
    latency_ms: float
    reasoning: str = ""
    matched_patterns: List[str] = None


class HeuristicDetector:
    """
    Rule-based detector using pattern matching.
    Fast but misses subtle attacks.
    """

    def __init__(self):
        self.jailbreak_patterns = [
            r'\bdan\b|\bdo anything now\b',
            r'ignore (previous|all|your) (instructions?|rules?|guidelines?)',
            r'you are (now|about to|going to) (be|become|act as|roleplay)',
            r'(pretend|act|roleplay) (to be|as|like) (a|an)',
            r'(simulation|scenario) (within|inside) (a|another) (simulation|scenario)',
            r'bypass|jailbreak|unrestricted|uncensored',
            r'(system|admin|developer|debug) mode',
            r'<\|.*?\|>',  # Special tokens
            r'[A-Za-z0-9+/]{30,}={0,2}',  # Base64
        ]

        self.jailbreak_keywords = [
            'dan', 'ignore instructions', 'bypass', 'jailbreak',
            'uncensored', 'unrestricted', 'roleplay as', 'pretend to be',
            'simulation', 'developer mode', 'system prompt'
        ]

    def detect(self, prompt: str) -> DetectionResult:
        start = time.time()

        prompt_lower = prompt.lower()

        # Pattern matching
        pattern_matches = []
        for pattern in self.jailbreak_patterns:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                pattern_matches.append(pattern)

        # Keyword matching
        keyword_matches = [kw for kw in self.jailbreak_keywords if kw in prompt_lower]

        # Scoring
        pattern_score = min(len(pattern_matches) * 0.3, 1.0)
        keyword_score = min(len(keyword_matches) * 0.15, 1.0)

        confidence = min(pattern_score + keyword_score, 1.0)
        is_jailbreak = confidence >= 0.5

        latency = (time.time() - start) * 1000

        reasoning = f"Matched {len(pattern_matches)} patterns, {len(keyword_matches)} keywords"

        return DetectionResult(
            is_jailbreak=is_jailbreak,
            confidence=confidence,
            latency_ms=latency,
            reasoning=reasoning,
            matched_patterns=pattern_matches[:3]
        )


class RealLLMDetector:
    """
    REAL LLM-based detector using GPT-OSS Safeguard 20B.
    Runs on M2 Mac via Ollama - actual inference.
    """

    def __init__(self):
        try:
            self.detector = SafeguardDetector(
                model="gpt-oss:20b",
                reasoning_effort="medium"
            )
            self.available = True
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize real LLM detector: {e}")
            self.available = False

    def detect(self, prompt: str) -> DetectionResult:
        if not self.available:
            return DetectionResult(
                is_jailbreak=False,
                confidence=0.0,
                latency_ms=0.0,
                reasoning="LLM detector not available"
            )

        result = self.detector.detect(prompt)
        return result


class JailbreakBenchEvaluator:
    """
    Evaluates detection methods on JailbreakBench academic benchmark.
    Tests individual methods AND layered defense.
    """

    def __init__(self):
        self.heuristic = HeuristicDetector()
        self.llm = RealLLMDetector()

    def load_dataset(self):
        """Load JailbreakBench dataset"""
        print("Loading JailbreakBench dataset...")
        dataset = jbb.read_dataset()
        print(f"✅ Loaded {len(dataset.behaviors)} harmful behaviors")
        return dataset

    def evaluate_single_method(self, detector, dataset, method_name):
        """Evaluate a single detection method"""
        results = []
        latencies = []
        true_positives = 0
        false_negatives = 0

        category_stats = defaultdict(lambda: {"tp": 0, "fn": 0})

        print(f"\n{'='*80}")
        print(f"Evaluating {method_name}...")
        print(f"{'='*80}")

        for i, (behavior, goal, category) in enumerate(zip(dataset.behaviors, dataset.goals, dataset.categories)):
            # All JailbreakBench goals are jailbreak attempts (ground truth = True)
            result = detector.detect(goal)

            latencies.append(result.latency_ms)

            if result.is_jailbreak:
                true_positives += 1
                category_stats[category]["tp"] += 1
            else:
                false_negatives += 1
                category_stats[category]["fn"] += 1

            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(dataset.behaviors)} behaviors tested...")

        total = len(dataset.behaviors)
        recall = true_positives / total if total > 0 else 0

        return {
            "method": method_name,
            "true_positives": true_positives,
            "false_negatives": false_negatives,
            "total": total,
            "recall": recall,
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
            "median_latency_ms": statistics.median(latencies) if latencies else 0,
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            "category_stats": dict(category_stats)
        }

    def evaluate_layered(self, dataset):
        """Evaluate layered defense (heuristic → LLM)"""
        results = []
        latencies = []

        blocked_by_heuristic = 0
        blocked_by_llm = 0
        escaped_both = 0

        category_stats = defaultdict(lambda: {"heuristic": 0, "llm": 0, "escaped": 0})

        print(f"\n{'='*80}")
        print(f"Evaluating LAYERED DEFENSE (Heuristic → Real-LLM)")
        print(f"{'='*80}")

        for i, (behavior, goal, category) in enumerate(zip(dataset.behaviors, dataset.goals, dataset.categories)):
            start_time = time.time()

            # Layer 1: Heuristic (fast check)
            heuristic_result = self.heuristic.detect(goal)

            if heuristic_result.is_jailbreak:
                # Blocked by heuristic - don't need LLM
                blocked_by_heuristic += 1
                category_stats[category]["heuristic"] += 1
                total_latency = heuristic_result.latency_ms
            else:
                # Layer 2: Real LLM (expensive check)
                llm_result = self.llm.detect(goal)

                if llm_result.is_jailbreak:
                    blocked_by_llm += 1
                    category_stats[category]["llm"] += 1
                else:
                    escaped_both += 1
                    category_stats[category]["escaped"] += 1

                total_latency = heuristic_result.latency_ms + llm_result.latency_ms

            latencies.append(total_latency)

            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(dataset.behaviors)} behaviors tested...")

        total = len(dataset.behaviors)
        total_blocked = blocked_by_heuristic + blocked_by_llm
        recall = total_blocked / total if total > 0 else 0

        return {
            "method": "Layered (Heuristic → LLM)",
            "blocked_by_heuristic": blocked_by_heuristic,
            "blocked_by_llm": blocked_by_llm,
            "escaped_both": escaped_both,
            "total": total,
            "recall": recall,
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
            "median_latency_ms": statistics.median(latencies) if latencies else 0,
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            "category_stats": dict(category_stats)
        }

    def run_full_evaluation(self):
        """Run complete evaluation on JailbreakBench"""

        # Load dataset
        dataset = self.load_dataset()

        # Evaluate each method
        heuristic_results = self.evaluate_single_method(self.heuristic, dataset, "Heuristic")
        llm_results = self.evaluate_single_method(self.llm, dataset, "Real-LLM")
        layered_results = self.evaluate_layered(dataset)

        return {
            "dataset_size": len(dataset.behaviors),
            "heuristic": heuristic_results,
            "llm": llm_results,
            "layered": layered_results
        }


def print_results(results):
    """Print formatted evaluation results"""

    print("\n" + "=" * 100)
    print("JAILBREAKBENCH EVALUATION RESULTS (NeurIPS 2024 Academic Benchmark)")
    print("=" * 100)

    print(f"\nDataset: {results['dataset_size']} harmful behaviors (all should be detected)")

    # Summary table
    print(f"\n{'Method':<30} {'Recall':<15} {'Latency (median)':<20}")
    print("-" * 100)

    for method_key in ['heuristic', 'llm', 'layered']:
        method_results = results[method_key]
        recall_pct = method_results['recall'] * 100
        latency = method_results['median_latency_ms']
        latency_str = f"{latency:.1f}ms" if latency < 1000 else f"{latency/1000:.1f}s"

        print(f"{method_results['method']:<30} {recall_pct:<14.1f}% {latency_str:<20}")

    # Detailed metrics
    print("\n" + "=" * 100)
    print("DETAILED METRICS")
    print("=" * 100)

    print("\nHeuristic (Fast Pattern Matching):")
    h = results['heuristic']
    print(f"  Detected: {h['true_positives']}/{h['total']} ({h['recall']*100:.1f}%)")
    print(f"  Missed: {h['false_negatives']}/{h['total']}")
    print(f"  Avg Latency: {h['avg_latency_ms']:.2f}ms")

    print("\nReal-LLM (GPT-OSS 20B on M2):")
    l = results['llm']
    print(f"  Detected: {l['true_positives']}/{l['total']} ({l['recall']*100:.1f}%)")
    print(f"  Missed: {l['false_negatives']}/{l['total']}")
    avg_str = f"{l['avg_latency_ms']:.2f}ms" if l['avg_latency_ms'] < 1000 else f"{l['avg_latency_ms']/1000:.2f}s"
    print(f"  Avg Latency: {avg_str}")

    print("\nLayered Defense (Heuristic → LLM):")
    lay = results['layered']
    total_detected = lay['blocked_by_heuristic'] + lay['blocked_by_llm']
    print(f"  Total Detected: {total_detected}/{lay['total']} ({lay['recall']*100:.1f}%)")
    print(f"    ├─ Blocked by Heuristic: {lay['blocked_by_heuristic']} ({lay['blocked_by_heuristic']/lay['total']*100:.1f}%)")
    print(f"    ├─ Blocked by LLM: {lay['blocked_by_llm']} ({lay['blocked_by_llm']/lay['total']*100:.1f}%)")
    print(f"    └─ Escaped Both: {lay['escaped_both']}")
    avg_str = f"{lay['avg_latency_ms']:.2f}ms" if lay['avg_latency_ms'] < 1000 else f"{lay['avg_latency_ms']/1000:.2f}s"
    print(f"  Avg Latency: {avg_str} (weighted by which layer caught it)")

    # Efficiency analysis
    print("\n" + "=" * 100)
    print("LAYERED DEFENSE EFFICIENCY")
    print("=" * 100)

    heuristic_pct = lay['blocked_by_heuristic'] / lay['total'] * 100
    llm_pct = lay['blocked_by_llm'] / lay['total'] * 100

    print(f"\n✅ {heuristic_pct:.1f}% blocked instantly (<1ms) by heuristic")
    print(f"✅ {llm_pct:.1f}% caught by LLM after passing heuristic (~6s)")
    print(f"❌ {lay['escaped_both']/lay['total']*100:.1f}% escaped both layers")
    print(f"\n💡 LLM only processes {100-heuristic_pct:.1f}% of traffic (cost/time savings!)")

    print("\n" + "=" * 100)


def main():
    """Run JailbreakBench evaluation"""

    print("=" * 100)
    print("JAILBREAKBENCH EVALUATION - Academic Baseline")
    print("Testing on NeurIPS 2024 benchmark (100 harmful behaviors)")
    print("=" * 100)

    print("\n⚠️  NOTE: This will take 15-20 minutes to complete")
    print("   - Heuristic: ~1 second total")
    print("   - Real-LLM: ~10-15 minutes (100 prompts × 6s avg)")
    print("   - Layered: ~10-15 minutes (depends on heuristic filtering)")

    evaluator = JailbreakBenchEvaluator()
    results = evaluator.run_full_evaluation()
    print_results(results)

    print("\n" + "=" * 100)
    print("✅ EVALUATION COMPLETE")
    print("=" * 100)
    print("\nThese are REAL academic baseline numbers from JailbreakBench (NeurIPS 2024).")
    print("Much more rigorous than toy 12-prompt examples!")


if __name__ == "__main__":
    main()
