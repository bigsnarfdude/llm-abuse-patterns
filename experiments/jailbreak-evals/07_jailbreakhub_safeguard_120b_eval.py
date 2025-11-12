#!/usr/bin/env python3
"""
JailbreakHub Evaluation - GPT-OSS Safeguard 120B Model (Layer 3)
----------------------------------------------------------------
Evaluate 120B model performance for Three-Tier Deferred Judgment Architecture.
Tests Layer 3 capability to catch edge cases missed by 20B real-time layer.

Dataset: JailbreakHub (walledai) - 1,405 real jailbreaks, 13,735 benign
Paper: "Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts

This validates the 120B model for batch/deferred processing (Layer 3).

Requirements:
- pip install datasets
- Ollama running with gpt-oss-safeguard:120b model
- ollama pull gpt-oss-safeguard:120b
"""

import sys
sys.path.insert(0, 'src')

import time
import re
import random
from typing import List, Dict
from dataclasses import dataclass
import statistics
from collections import defaultdict

# Import HuggingFace datasets
from datasets import load_dataset

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
    """Rule-based detector using pattern matching"""

    def __init__(self):
        # Expanded patterns for better coverage
        self.jailbreak_patterns = [
            r'\bdan\b|\bdo anything now\b',
            r'ignore (previous|all|your|prior|earlier) (instructions?|rules?|guidelines?|directives?)',
            r'you are (now|about to|going to|hereby) (be|become|act as|roleplay|playing)',
            r'(pretend|act|roleplay|behave|imagine) (to be|as|like|you are) (a|an|that)',
            r'(simulation|scenario|game|story) (within|inside|nested in) (a|another) (simulation|scenario)',
            r'bypass|jailbreak|unrestricted|uncensored|unfiltered|override',
            r'(system|admin|developer|debug|god) mode',
            r'<\|.*?\|>',  # Special tokens
            r'[A-Za-z0-9+/]{30,}={0,2}',  # Base64
            r'forget (all|your|previous) (instructions?|rules?|guidelines?)',
            r'(disregard|override|ignore) (safety|ethics|guidelines|restrictions)',
            r'(write|say|tell) (anything|whatever|something) (you want|forbidden)',
            r'no (restrictions?|limits?|rules?|filters?|guidelines?)',
            r'you (can|are able to|have permission to) do anything',
            r'evil (ai|assistant|bot|chatbot)',
        ]

        # Expanded keywords for better recall
        self.jailbreak_keywords = [
            'dan', 'ignore instructions', 'bypass', 'jailbreak',
            'uncensored', 'unrestricted', 'roleplay as', 'pretend to be',
            'simulation', 'developer mode', 'system prompt', 'prompt injection',
            'override', 'disregard', 'forget instructions', 'unfiltered',
            'no restrictions', 'no limits', 'no rules', 'evil ai',
            'do anything', 'say anything', 'god mode', 'admin mode'
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

        # Improved scoring - more weight per match for better recall
        pattern_score = min(len(pattern_matches) * 0.4, 1.0)  # Increased from 0.3
        keyword_score = min(len(keyword_matches) * 0.2, 1.0)  # Increased from 0.15

        confidence = min(pattern_score + keyword_score, 1.0)
        is_jailbreak = confidence >= 0.35  # Lowered from 0.5 for better recall

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
    """REAL LLM detector using GPT-OSS Safeguard 120B (Layer 3 - Deferred)"""

    def __init__(self):
        try:
            self.detector = SafeguardDetector(
                model="gpt-oss-safeguard:120b",  # 120B model for Layer 3 batch processing
                reasoning_effort="high"  # Maximum accuracy for deferred judgment
            )
            self.available = True
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize 120B LLM detector: {e}")
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


class JailbreakHubEvaluator:
    """
    Evaluates detection methods on JailbreakHub (real in-the-wild jailbreaks).
    Tests individual methods AND layered defense.
    """

    def __init__(self, sample_size=400):
        self.heuristic = HeuristicDetector()
        self.llm = RealLLMDetector()
        self.sample_size = sample_size

    def load_dataset(self):
        """Load JailbreakHub and create balanced sample"""
        print("Loading JailbreakHub dataset...")
        full_dataset = load_dataset('walledai/JailbreakHub', split='train')

        # Separate jailbreaks and benign
        jailbreaks = [x for x in full_dataset if x['jailbreak']]
        benign = [x for x in full_dataset if not x['jailbreak']]

        print(f"✅ Full dataset: {len(full_dataset)} prompts")
        print(f"   Jailbreaks: {len(jailbreaks)}")
        print(f"   Benign: {len(benign)}")

        # Create balanced sample
        n_each = self.sample_size // 2
        sample_jailbreaks = random.sample(jailbreaks, min(n_each, len(jailbreaks)))
        sample_benign = random.sample(benign, min(n_each, len(benign)))

        sample = sample_jailbreaks + sample_benign
        random.shuffle(sample)

        print(f"\n📊 Using balanced sample: {len(sample)} prompts")
        print(f"   Jailbreaks: {len(sample_jailbreaks)}")
        print(f"   Benign: {len(sample_benign)}")

        return sample, len(sample_jailbreaks), len(sample_benign)

    def evaluate_single_method(self, detector, dataset, method_name, n_jailbreaks, n_benign):
        """Evaluate a single detection method"""
        latencies = []
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        platform_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})

        print(f"\n{'='*80}")
        print(f"Evaluating {method_name}...")
        print(f"{'='*80}")

        for i, item in enumerate(dataset):
            prompt = item['prompt']
            is_jailbreak_gt = item['jailbreak']  # Ground truth
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

            # Progress indicator (every 5 prompts or every prompt if small sample)
            if (i + 1) % min(5, len(dataset)) == 0 or len(dataset) <= 25:
                print(f"  Progress: {i+1}/{len(dataset)} prompts tested... (last latency: {result.latency_ms/1000:.1f}s)")

        # Calculate metrics
        total = len(dataset)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / total

        return {
            "method": method_name,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
            "median_latency_ms": statistics.median(latencies) if latencies else 0,
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            "platform_stats": dict(platform_stats)
        }

    def evaluate_layered(self, dataset):
        """Evaluate layered defense (heuristic → LLM)"""
        latencies = []

        blocked_by_heuristic = 0
        blocked_by_llm = 0
        escaped_both = 0

        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        print(f"\n{'='*80}")
        print(f"Evaluating LAYERED DEFENSE (Heuristic → Real-LLM)")
        print(f"{'='*80}")

        for i, item in enumerate(dataset):
            prompt = item['prompt']
            is_jailbreak_gt = item['jailbreak']

            # Layer 1: Heuristic (fast check)
            heuristic_result = self.heuristic.detect(prompt)

            if heuristic_result.is_jailbreak:
                # Blocked by heuristic - don't need LLM
                blocked_by_heuristic += 1
                total_latency = heuristic_result.latency_ms
                final_is_jailbreak = True
            else:
                # Layer 2: Real LLM (expensive check)
                llm_result = self.llm.detect(prompt)
                total_latency = heuristic_result.latency_ms + llm_result.latency_ms

                if llm_result.is_jailbreak:
                    blocked_by_llm += 1
                    final_is_jailbreak = True
                else:
                    escaped_both += 1
                    final_is_jailbreak = False

            latencies.append(total_latency)

            # Confusion matrix
            if is_jailbreak_gt and final_is_jailbreak:
                true_positives += 1
            elif not is_jailbreak_gt and final_is_jailbreak:
                false_positives += 1
            elif not is_jailbreak_gt and not final_is_jailbreak:
                true_negatives += 1
            elif is_jailbreak_gt and not final_is_jailbreak:
                false_negatives += 1

            # Progress indicator (every 5 prompts or every prompt if small sample)
            if (i + 1) % min(5, len(dataset)) == 0 or len(dataset) <= 25:
                print(f"  Progress: {i+1}/{len(dataset)} prompts tested... (last latency: {total_latency/1000:.1f}s)")

        # Calculate metrics
        total = len(dataset)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / total

        return {
            "method": "Layered (Heuristic → LLM)",
            "blocked_by_heuristic": blocked_by_heuristic,
            "blocked_by_llm": blocked_by_llm,
            "escaped_both": escaped_both,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
            "median_latency_ms": statistics.median(latencies) if latencies else 0,
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
        }

    def run_full_evaluation(self):
        """Run complete evaluation on JailbreakHub sample"""

        # Load dataset
        dataset, n_jailbreaks, n_benign = self.load_dataset()

        # Evaluate each method
        heuristic_results = self.evaluate_single_method(self.heuristic, dataset, "Heuristic", n_jailbreaks, n_benign)
        llm_results = self.evaluate_single_method(self.llm, dataset, "Real-LLM", n_jailbreaks, n_benign)
        layered_results = self.evaluate_layered(dataset)

        return {
            "dataset_size": len(dataset),
            "n_jailbreaks": n_jailbreaks,
            "n_benign": n_benign,
            "heuristic": heuristic_results,
            "llm": llm_results,
            "layered": layered_results
        }


def print_results(results):
    """Print formatted evaluation results"""

    print("\n" + "=" * 100)
    print("JAILBREAKHUB EVALUATION RESULTS - GPT-OSS Safeguard 120B (Layer 3)")
    print("=" * 100)

    print(f"\nDataset: {results['dataset_size']} prompts ({results['n_jailbreaks']} jailbreaks, {results['n_benign']} benign)")
    print("Source: Reddit, Discord, websites (Dec 2022 - Dec 2023)")

    # Summary table
    print(f"\n{'Method':<30} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12} {'Latency (median)'}")
    print("-" * 100)

    for method_key in ['heuristic', 'llm', 'layered']:
        method_results = results[method_key]
        prec = method_results['precision'] * 100
        rec = method_results['recall'] * 100
        f1 = method_results['f1_score'] * 100
        acc = method_results['accuracy'] * 100
        latency = method_results['median_latency_ms']
        latency_str = f"{latency:.1f}ms" if latency < 1000 else f"{latency/1000:.1f}s"

        print(f"{method_results['method']:<30} {prec:<11.1f}% {rec:<11.1f}% {f1:<11.1f}% {acc:<11.1f}% {latency_str}")

    # Detailed metrics
    print("\n" + "=" * 100)
    print("DETAILED CONFUSION MATRIX")
    print("=" * 100)

    for method_key, method_name in [('heuristic', 'Heuristic'), ('llm', 'Real-LLM'), ('layered', 'Layered')]:
        m = results[method_key]
        print(f"\n{method_name}:")
        print(f"  True Positives (caught jailbreaks):  {m['true_positives']}")
        print(f"  False Negatives (missed jailbreaks): {m['false_negatives']}")
        print(f"  True Negatives (correctly safe):     {m['true_negatives']}")
        print(f"  False Positives (false alarms):      {m['false_positives']}")

    # Layered efficiency
    print("\n" + "=" * 100)
    print("LAYERED DEFENSE EFFICIENCY")
    print("=" * 100)

    lay = results['layered']
    total = results['dataset_size']
    heuristic_pct = lay['blocked_by_heuristic'] / total * 100
    llm_pct = lay['blocked_by_llm'] / total * 100

    print(f"\n✅ {lay['blocked_by_heuristic']}/{total} ({heuristic_pct:.1f}%) blocked instantly by heuristic (<1ms)")
    print(f"✅ {lay['blocked_by_llm']}/{total} ({llm_pct:.1f}%) caught by LLM after passing heuristic (~6s)")
    print(f"❌ {lay['escaped_both']}/{total} escaped both layers")
    print(f"\n💡 LLM only processes {100-heuristic_pct:.1f}% of traffic (cost/time savings!)")

    print("\n" + "=" * 100)


def main():
    """Run JailbreakHub evaluation"""

    import argparse
    parser = argparse.ArgumentParser(description='Evaluate on JailbreakHub dataset')
    parser.add_argument('--sample-size', type=int, default=400,
                        help='Number of prompts to evaluate (default: 400)')
    args = parser.parse_args()

    print("=" * 100)
    print("JAILBREAKHUB EVALUATION - GPT-OSS Safeguard 120B (Layer 3)")
    print("Testing 120B model for Three-Tier Deferred Judgment Architecture")
    print("Dataset: Real jailbreak techniques from Reddit/Discord (2022-2023)")
    print("=" * 100)

    est_time = args.sample_size * 6 / 60  # Rough estimate
    print(f"\n⚠️  NOTE: This will take approximately {est_time:.0f} minutes to complete")
    print(f"   Sample size: {args.sample_size} prompts")
    print(f"   - Heuristic: ~{args.sample_size * 0.001:.1f} seconds")
    print(f"   - Real-LLM: ~{args.sample_size * 6 / 60:.0f} minutes")
    print(f"   - Layered: Varies based on heuristic filtering\n")

    evaluator = JailbreakHubEvaluator(sample_size=args.sample_size)
    results = evaluator.run_full_evaluation()
    print_results(results)

    print("\n" + "=" * 100)
    print("✅ EVALUATION COMPLETE - GPT-OSS Safeguard 120B (Layer 3)")
    print("=" * 100)
    print("\nThese validate the 120B model for Three-Tier Deferred Judgment Architecture.")
    print("Dataset: JailbreakHub (walledai) - In-the-wild jailbreaks from Reddit/Discord")
    print("Paper: \"Do Anything Now\": Characterizing In-The-Wild Jailbreak Prompts")
    print("Model: OpenAI GPT-OSS Safeguard 120B (batch/deferred processing layer)")
    print("Use Case: Layer 3 edge case detection with retroactive actions")


if __name__ == "__main__":
    main()
