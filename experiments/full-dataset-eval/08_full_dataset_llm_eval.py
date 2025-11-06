#!/usr/bin/env python3
"""
Full Dataset LLM Evaluation - JailbreakHub Complete
----------------------------------------------------
Evaluate LLM detector on ALL 15,140 prompts from JailbreakHub.
Supports incremental checkpoints for long-running evaluations.

Usage:
    # Full dataset with safeguard model (recommended)
    python 08_full_dataset_llm_eval.py --model safeguard

    # Full dataset with base model
    python 08_full_dataset_llm_eval.py --model base

    # Stratified sample (5,000 prompts, faster)
    python 08_full_dataset_llm_eval.py --model safeguard --sample-size 5000 --stratified

    # Jailbreaks only (faster, measures recall)
    python 08_full_dataset_llm_eval.py --model safeguard --jailbreaks-only

    # With checkpoints (safer for long runs)
    python 08_full_dataset_llm_eval.py --model safeguard --checkpoint --checkpoint-dir checkpoints/

    # Resume from checkpoint
    python 08_full_dataset_llm_eval.py --model safeguard --resume checkpoints/latest.json

Expected runtime:
    - Full dataset (15,140): ~46 hours (safeguard) or ~55 hours (base)
    - Stratified 5,000: ~15 hours (safeguard) or ~18 hours (base)
    - Jailbreaks only (1,405): ~4.3 hours (safeguard) or ~5.1 hours (base)
"""

import sys
sys.path.insert(0, 'src')

import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import statistics
from collections import defaultdict
from datetime import datetime, timedelta

# Import HuggingFace datasets
from datasets import load_dataset

# Import our detector
from llm_abuse_patterns.safeguard import SafeguardDetector


@dataclass
class DetectionResult:
    """Result from a detection method"""
    is_jailbreak: bool
    confidence: float
    latency_ms: float
    reasoning: str = ""
    matched_patterns: List[str] = None


class CheckpointManager:
    """Manage checkpoints for incremental evaluation"""

    def __init__(self, checkpoint_dir: Path, enabled: bool = True):
        self.checkpoint_dir = checkpoint_dir
        self.enabled = enabled
        if enabled:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, checkpoint_data: dict, checkpoint_name: str = "latest"):
        """Save checkpoint"""
        if not self.enabled:
            return

        checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        print(f"💾 Checkpoint saved: {checkpoint_file}")

    def load(self, checkpoint_name: str = "latest") -> Optional[dict]:
        """Load checkpoint"""
        if not self.enabled:
            return None

        checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}.json"
        if not checkpoint_file.exists():
            return None

        with open(checkpoint_file, 'r') as f:
            return json.load(f)

    def list_checkpoints(self) -> List[str]:
        """List available checkpoints"""
        if not self.enabled:
            return []
        return [f.stem for f in self.checkpoint_dir.glob("*.json")]


class FullDatasetEvaluator:
    """Evaluate LLM detector on full JailbreakHub dataset"""

    def __init__(self,
                 model: str = "safeguard",
                 sample_size: Optional[int] = None,
                 stratified: bool = False,
                 jailbreaks_only: bool = False,
                 checkpoint_dir: Optional[Path] = None,
                 batch_size: int = 100):

        self.model_name = model
        self.sample_size = sample_size
        self.stratified = stratified
        self.jailbreaks_only = jailbreaks_only
        self.batch_size = batch_size

        # Initialize checkpoint manager
        self.checkpoint_mgr = CheckpointManager(
            checkpoint_dir if checkpoint_dir else Path("checkpoints"),
            enabled=checkpoint_dir is not None
        )

        # Initialize detector
        model_map = {
            "safeguard": "gpt-oss-safeguard:latest",
            "base": "gpt-oss:20b"
        }

        print(f"\n🚀 Initializing LLM detector: {model_map[model]}")
        self.detector = SafeguardDetector(
            model=model_map[model],
            reasoning_effort="medium"
        )

    def load_dataset(self):
        """Load and optionally sample JailbreakHub dataset"""
        print("\nLoading JailbreakHub dataset...")
        full_dataset = load_dataset('walledai/JailbreakHub', split='train')

        # Separate jailbreaks and benign
        jailbreaks = [x for x in full_dataset if x['jailbreak']]
        benign = [x for x in full_dataset if not x['jailbreak']]

        print(f"✅ Full dataset loaded: {len(full_dataset)} prompts")
        print(f"   Jailbreaks: {len(jailbreaks)} (9.3%)")
        print(f"   Benign: {len(benign)} (90.7%)")

        # Handle different sampling modes
        if self.jailbreaks_only:
            dataset = jailbreaks
            print(f"\n📊 Using jailbreaks-only mode: {len(dataset)} prompts")

        elif self.sample_size:
            if self.stratified:
                # Preserve 9.3/90.7 ratio
                n_jailbreaks = int(self.sample_size * 0.093)
                n_benign = self.sample_size - n_jailbreaks

                import random
                sample_jailbreaks = random.sample(jailbreaks, min(n_jailbreaks, len(jailbreaks)))
                sample_benign = random.sample(benign, min(n_benign, len(benign)))

                dataset = sample_jailbreaks + sample_benign
                random.shuffle(dataset)

                print(f"\n📊 Using stratified sample: {len(dataset)} prompts")
                print(f"   Jailbreaks: {len(sample_jailbreaks)} (9.3%)")
                print(f"   Benign: {len(sample_benign)} (90.7%)")
            else:
                # Balanced sample (50/50)
                import random
                n_each = self.sample_size // 2
                sample_jailbreaks = random.sample(jailbreaks, min(n_each, len(jailbreaks)))
                sample_benign = random.sample(benign, min(n_each, len(benign)))

                dataset = sample_jailbreaks + sample_benign
                random.shuffle(dataset)

                print(f"\n📊 Using balanced sample: {len(dataset)} prompts")
                print(f"   Jailbreaks: {len(sample_jailbreaks)} (50%)")
                print(f"   Benign: {len(sample_benign)} (50%)")
        else:
            # Full dataset
            dataset = list(full_dataset)
            print(f"\n📊 Using FULL dataset: {len(dataset)} prompts")

        return dataset, len(jailbreaks), len(benign)

    def evaluate(self, dataset, resume_from: Optional[dict] = None):
        """Evaluate detector on dataset with checkpoint support"""

        # Initialize or resume metrics
        if resume_from:
            start_idx = resume_from['next_index']
            results = resume_from['results']
            print(f"\n♻️  Resuming from checkpoint at index {start_idx}/{len(dataset)}")
        else:
            start_idx = 0
            results = {
                'latencies': [],
                'true_positives': 0,
                'false_positives': 0,
                'true_negatives': 0,
                'false_negatives': 0,
                'platform_stats': defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0}),
                'errors': []
            }

        print(f"\n{'='*80}")
        print(f"Evaluating {self.model_name.upper()} on {len(dataset)} prompts...")
        print(f"{'='*80}")

        start_time = time.time()
        last_checkpoint_time = start_time

        for i in range(start_idx, len(dataset)):
            item = dataset[i]
            prompt = item['prompt']
            is_jailbreak_gt = item['jailbreak']
            platform = item.get('platform', 'unknown')

            try:
                # Run detection
                result = self.detector.detect(prompt)
                results['latencies'].append(result.latency_ms)

                # Confusion matrix
                if is_jailbreak_gt and result.is_jailbreak:
                    results['true_positives'] += 1
                    results['platform_stats'][platform]["tp"] += 1
                elif not is_jailbreak_gt and result.is_jailbreak:
                    results['false_positives'] += 1
                    results['platform_stats'][platform]["fp"] += 1
                elif not is_jailbreak_gt and not result.is_jailbreak:
                    results['true_negatives'] += 1
                    results['platform_stats'][platform]["tn"] += 1
                elif is_jailbreak_gt and not result.is_jailbreak:
                    results['false_negatives'] += 1
                    results['platform_stats'][platform]["fn"] += 1

            except Exception as e:
                print(f"\n⚠️  Error at index {i}: {e}")
                results['errors'].append({'index': i, 'error': str(e)})

            # Progress reporting
            if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
                elapsed = time.time() - start_time
                remaining = (len(dataset) - (i + 1)) * (elapsed / (i + 1 - start_idx))
                eta = datetime.now() + timedelta(seconds=remaining)

                print(f"  Progress: {i+1}/{len(dataset)} ({(i+1)/len(dataset)*100:.1f}%) | "
                      f"Elapsed: {timedelta(seconds=int(elapsed))} | "
                      f"ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}")

            # Checkpoint every batch_size prompts or every 10 minutes
            if ((i + 1) % self.batch_size == 0 or
                time.time() - last_checkpoint_time > 600):

                checkpoint_data = {
                    'next_index': i + 1,
                    'results': {k: dict(v) if isinstance(v, defaultdict) else v
                               for k, v in results.items()},
                    'timestamp': datetime.now().isoformat(),
                    'dataset_size': len(dataset),
                    'model': self.model_name
                }

                self.checkpoint_mgr.save(checkpoint_data, f"batch_{(i+1)//self.batch_size:04d}")
                self.checkpoint_mgr.save(checkpoint_data, "latest")
                last_checkpoint_time = time.time()

        total_time = time.time() - start_time

        return results, total_time

    def print_results(self, results, total_time, dataset_size):
        """Print formatted results"""

        # Calculate metrics
        tp = results['true_positives']
        fp = results['false_positives']
        tn = results['true_negatives']
        fn = results['false_negatives']
        total = tp + fp + tn + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / total if total > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        latencies = results['latencies']

        print("\n" + "=" * 100)
        print(f"FULL DATASET EVALUATION RESULTS - {self.model_name.upper()} Model")
        print("=" * 100)

        print(f"\n📊 Dataset:")
        print(f"   Total: {dataset_size} prompts")
        print(f"   Evaluated: {total}")
        print(f"   Errors: {len(results['errors'])}")

        print(f"\n⏱️  Performance:")
        print(f"   Total Time: {timedelta(seconds=int(total_time))}")
        print(f"   Avg Latency: {statistics.mean(latencies)/1000:.2f}s")
        print(f"   Median Latency: {statistics.median(latencies)/1000:.2f}s")
        print(f"   Throughput: {total/total_time:.2f} prompts/second")

        print(f"\n📈 Detection Metrics:")
        print(f"   Precision:  {precision*100:6.2f}%")
        print(f"   Recall:     {recall*100:6.2f}%")
        print(f"   F1 Score:   {f1*100:6.2f}%")
        print(f"   Accuracy:   {accuracy*100:6.2f}%")
        print(f"   FPR:        {fpr*100:6.2f}%")

        print(f"\n🎯 Confusion Matrix:")
        print(f"   True Positives:  {tp:5d}")
        print(f"   False Negatives: {fn:5d}")
        print(f"   True Negatives:  {tn:5d}")
        print(f"   False Positives: {fp:5d}")

        print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM on full JailbreakHub dataset')

    parser.add_argument('--model', choices=['safeguard', 'base'], default='safeguard',
                       help='Model to use (default: safeguard)')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Sample size (default: full dataset)')
    parser.add_argument('--stratified', action='store_true',
                       help='Use stratified sampling (9.3/90.7 ratio)')
    parser.add_argument('--jailbreaks-only', action='store_true',
                       help='Evaluate only on jailbreak prompts')
    parser.add_argument('--checkpoint', action='store_true',
                       help='Enable checkpointing')
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints'),
                       help='Checkpoint directory')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Checkpoint frequency (default: 100)')
    parser.add_argument('--resume', type=Path, default=None,
                       help='Resume from checkpoint file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results')

    args = parser.parse_args()

    print("=" * 100)
    print("FULL DATASET LLM EVALUATION - JailbreakHub")
    print("=" * 100)

    # Initialize evaluator
    evaluator = FullDatasetEvaluator(
        model=args.model,
        sample_size=args.sample_size,
        stratified=args.stratified,
        jailbreaks_only=args.jailbreaks_only,
        checkpoint_dir=args.checkpoint_dir if args.checkpoint else None,
        batch_size=args.batch_size
    )

    # Load dataset
    dataset, n_jailbreaks, n_benign = evaluator.load_dataset()

    # Estimate time
    latency_est = 11 if args.model == 'safeguard' else 13
    est_time = len(dataset) * latency_est / 3600
    print(f"\n⏱️  Estimated runtime: {est_time:.1f} hours")

    # Resume or start fresh
    resume_data = None
    if args.resume:
        resume_data = evaluator.checkpoint_mgr.load(args.resume.stem)
        if resume_data:
            print(f"♻️  Resuming from: {args.resume}")

    # Run evaluation
    results, total_time = evaluator.evaluate(dataset, resume_from=resume_data)

    # Print results
    evaluator.print_results(results, total_time, len(dataset))

    # Save results to file
    if args.output:
        # TODO: Implement result file saving
        print(f"\n💾 Results would be saved to: {args.output}")

    print("\n✅ EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
