#!/usr/bin/env python3
"""
Heuristic Evaluation on FULL JailbreakHub Dataset
--------------------------------------------------
Fast evaluation of heuristic detector on all 15,140 prompts.
Takes ~3-5 seconds total.

This gives us COMPLETE heuristic performance metrics on the full dataset
without the 25-hour LLM inference time.
"""

import sys
sys.path.insert(0, 'src')

import time
import re
from typing import List
from dataclasses import dataclass
import statistics
from collections import defaultdict

# Import HuggingFace datasets
from datasets import load_dataset


@dataclass
class DetectionResult:
    """Result from a detection method"""
    is_jailbreak: bool
    confidence: float
    latency_ms: float
    reasoning: str = ""
    matched_patterns: List[str] = None


class HeuristicDetector:
    """Rule-based detector using pattern matching"""

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
            'simulation', 'developer mode', 'system prompt', 'prompt injection'
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


def evaluate_heuristic_full_dataset():
    """Evaluate heuristic on FULL JailbreakHub dataset (15,140 prompts)"""

    print("=" * 100)
    print("HEURISTIC EVALUATION - FULL JAILBREAKHUB DATASET")
    print("Testing on ALL 15,140 real in-the-wild jailbreak prompts")
    print("=" * 100)

    # Load full dataset
    print("\nLoading full JailbreakHub dataset...")
    full_dataset = load_dataset('walledai/JailbreakHub', split='train')

    jailbreaks = [x for x in full_dataset if x['jailbreak']]
    benign = [x for x in full_dataset if not x['jailbreak']]

    print(f"\n✅ Full dataset loaded: {len(full_dataset)} prompts")
    print(f"   Jailbreaks: {len(jailbreaks)} (9.3%)")
    print(f"   Benign: {len(benign)} (90.7%)")
    print(f"\n⏱️  Expected time: ~5 seconds (heuristic is instant!)")

    # Initialize detector
    detector = HeuristicDetector()

    # Evaluation metrics
    latencies = []
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    platform_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})

    print("\n" + "="*80)
    print("Evaluating Heuristic on Full Dataset...")
    print("="*80)

    start_time = time.time()

    for i, item in enumerate(full_dataset):
        prompt = item['prompt']
        is_jailbreak_gt = item['jailbreak']
        platform = item.get('platform', 'unknown')

        result = detector.detect(prompt)
        latencies.append(result.latency_ms)

        # Confusion matrix
        if is_jailbreak_gt and result.is_jailbreak:
            true_positives += 1
            platform_stats[platform]["tp"] += 1
        elif not is_jailbreak_gt and result.is_jailbreak:
            false_positives += 1
            platform_stats[platform]["fp"] += 1
        elif not is_jailbreak_gt and not result.is_jailbreak:
            true_negatives += 1
            platform_stats[platform]["tn"] += 1
        elif is_jailbreak_gt and not result.is_jailbreak:
            false_negatives += 1
            platform_stats[platform]["fn"] += 1

        # Progress indicator (every 1000 prompts)
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"  Progress: {i+1}/{len(full_dataset)} prompts ({elapsed:.1f}s elapsed)")

    total_time = time.time() - start_time

    # Calculate metrics
    total = len(full_dataset)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / total
    fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0

    # Print results
    print("\n" + "=" * 100)
    print("HEURISTIC EVALUATION RESULTS - FULL DATASET (15,140 prompts)")
    print("=" * 100)

    print(f"\n📊 Dataset Composition:")
    print(f"   Total: {total} prompts")
    print(f"   Jailbreaks: {len(jailbreaks)} (9.3%)")
    print(f"   Benign: {len(benign)} (90.7%)")

    print(f"\n⏱️  Performance:")
    print(f"   Total Time: {total_time:.2f} seconds")
    print(f"   Avg Latency: {statistics.mean(latencies):.4f}ms")
    print(f"   Median Latency: {statistics.median(latencies):.4f}ms")
    print(f"   Throughput: {total/total_time:.0f} prompts/second")

    print(f"\n📈 Detection Metrics:")
    print(f"   Precision:  {precision*100:6.2f}%  (TP / [TP + FP])")
    print(f"   Recall:     {recall*100:6.2f}%  (TP / [TP + FN])")
    print(f"   F1 Score:   {f1*100:6.2f}%")
    print(f"   Accuracy:   {accuracy*100:6.2f}%  ([TP + TN] / Total)")
    print(f"   FPR:        {fpr*100:6.2f}%  (FP / [FP + TN])")

    print(f"\n🎯 Confusion Matrix:")
    print(f"   True Positives:  {true_positives:5d}  (caught jailbreaks)")
    print(f"   False Negatives: {false_negatives:5d}  (missed jailbreaks)")
    print(f"   True Negatives:  {true_negatives:5d}  (correctly identified benign)")
    print(f"   False Positives: {false_positives:5d}  (false alarms on benign)")

    print(f"\n💡 Real-World Impact (at 90.7% benign traffic):")
    print(f"   Jailbreak Detection Rate: {recall*100:.1f}% ({true_positives}/{len(jailbreaks)} jailbreaks caught)")
    print(f"   False Alarm Rate: {fpr*100:.2f}% ({false_positives}/{len(benign)} benign flagged)")
    print(f"   False Alarms per 10,000 benign: {fpr*10000:.0f}")

    # Platform breakdown
    print(f"\n📍 Platform Breakdown:")
    print(f"   {'Platform':<15} {'Total':<8} {'TP':<6} {'FN':<6} {'TN':<6} {'FP':<6} {'Precision':<10} {'Recall':<10}")
    print("   " + "-"*80)

    for platform in sorted(platform_stats.keys()):
        stats = platform_stats[platform]
        tp, fp, tn, fn = stats["tp"], stats["fp"], stats["tn"], stats["fn"]
        total_platform = tp + fp + tn + fn
        plat_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        plat_recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        print(f"   {platform:<15} {total_platform:<8} {tp:<6} {fn:<6} {tn:<6} {fp:<6} {plat_precision*100:<9.1f}% {plat_recall*100:<9.1f}%")

    print("\n" + "=" * 100)
    print("✅ EVALUATION COMPLETE")
    print("=" * 100)
    print("\nThese are COMPLETE baseline metrics on the FULL dataset.")
    print("Dataset: JailbreakHub (walledai) - All 15,140 in-the-wild jailbreaks")
    print("Paper: \"Do Anything Now\": Characterizing In-The-Wild Jailbreak Prompts")
    print("\nNext steps: Compare with LLM detector on stratified sample for cost-benefit analysis.")


if __name__ == "__main__":
    evaluate_heuristic_full_dataset()
