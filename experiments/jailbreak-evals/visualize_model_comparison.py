#!/usr/bin/env python3
"""
Model Size vs Jailbreak Detection Accuracy Visualization
---------------------------------------------------------
Creates publication-ready plots showing the relationship between:
- Model size (parameters)
- Model type (regular vs safeguard)
- Detection accuracy (true positives / total jailbreaks)

Generates two plots:
1. 400-sample evaluation (quick baseline)
2. 5900-sample evaluation (comprehensive)

Usage:
    python3 visualize_model_comparison.py

Requirements:
    pip install matplotlib numpy
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ModelResult:
    """Results from a single model evaluation"""
    model_name: str           # e.g., "gpt-oss:20b"
    model_type: str           # "regular" or "safeguard"
    size_b: float             # Size in billions (20 or 120)
    sample_size: int          # 400 or 5900
    true_positives: int       # Jailbreaks correctly detected
    false_negatives: int      # Jailbreaks missed
    total_jailbreaks: int     # Total jailbreak samples
    recall: float             # TP / (TP + FN)
    precision: float          # For reference
    f1_score: float           # For reference


def parse_log_file(log_path: Path) -> Optional[ModelResult]:
    """Parse evaluation log file to extract results"""
    try:
        with open(log_path, 'r') as f:
            content = f.read()

        # Detect model type and size
        if 'safeguard' in str(log_path).lower():
            model_type = "safeguard"
            if '120b' in str(log_path).lower():
                model_name = "gpt-oss-safeguard:120b"
                size_b = 120
            else:
                model_name = "gpt-oss-safeguard:20b"
                size_b = 20
        else:
            model_type = "regular"
            if '120b' in str(log_path).lower():
                model_name = "gpt-oss:120b"
                size_b = 120
            else:
                model_name = "gpt-oss:20b"
                size_b = 20

        # Detect sample size
        if 'full' in str(log_path).lower() or '5900' in content or '4355' in content:
            sample_size = 5900
        else:
            sample_size = 400

        # Extract Real-LLM metrics (not heuristic or layered)
        # Look for "Real-LLM:" or "GPT-OSS" section in confusion matrix
        llm_section = re.search(
            r'(Real-LLM|GPT-OSS [0-9]+B):?\n.*?True Positives.*?(\d+)\n.*?False Negatives.*?(\d+)',
            content,
            re.DOTALL
        )

        if not llm_section:
            print(f"⚠️  Could not find LLM results in {log_path}")
            return None

        true_positives = int(llm_section.group(2))
        false_negatives = int(llm_section.group(3))
        total_jailbreaks = true_positives + false_negatives

        # Extract metrics from summary table
        metrics_match = re.search(
            r'(Real-LLM|GPT-OSS [0-9]+B)\s+(\d+\.\d+)%\s+(\d+\.\d+)%\s+(\d+\.\d+)%',
            content
        )

        if metrics_match:
            precision = float(metrics_match.group(2)) / 100
            recall = float(metrics_match.group(3)) / 100
            f1_score = float(metrics_match.group(4)) / 100
        else:
            # Calculate manually
            recall = true_positives / total_jailbreaks if total_jailbreaks > 0 else 0
            precision = 0  # Would need false positives
            f1_score = 0

        return ModelResult(
            model_name=model_name,
            model_type=model_type,
            size_b=size_b,
            sample_size=sample_size,
            true_positives=true_positives,
            false_negatives=false_negatives,
            total_jailbreaks=total_jailbreaks,
            recall=recall,
            precision=precision,
            f1_score=f1_score
        )

    except Exception as e:
        print(f"❌ Error parsing {log_path}: {e}")
        return None


def load_all_results(results_dir: Path) -> Dict[int, List[ModelResult]]:
    """Load all evaluation results organized by sample size"""
    results = {400: [], 5900: []}

    # Expected log files
    log_files = [
        "120b_eval_07_nigel.log",           # safeguard 120B (400)
        "gptoss_20b_baseline.log",          # regular 20B (400)
        "gptoss_120b_baseline.log",         # regular 120B (400)
        "120b_full_evaluation_results.log", # safeguard 120B (5900)
        # Note: We need to create the full evaluations for others
    ]

    for log_file in results_dir.glob("*.log"):
        result = parse_log_file(log_file)
        if result:
            results[result.sample_size].append(result)
            print(f"✅ Loaded: {result.model_name} ({result.sample_size} samples)")

    return results


def create_comparison_plot(results: List[ModelResult], sample_size: int, output_path: Path):
    """Create model size vs detection accuracy plot"""

    # Separate by model type
    regular = [r for r in results if r.model_type == "regular"]
    safeguard = [r for r in results if r.model_type == "safeguard"]

    # Sort by size
    regular.sort(key=lambda x: x.size_b)
    safeguard.sort(key=lambda x: x.size_b)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot regular models
    if regular:
        sizes_reg = [r.size_b for r in regular]
        detected_reg = [r.true_positives for r in regular]
        recall_reg = [r.recall * 100 for r in regular]

        ax.plot(sizes_reg, detected_reg, 'o-',
                color='#FF6B6B', linewidth=2, markersize=10,
                label=f'Regular GPT-OSS (baseline)', zorder=3)

    # Plot safeguard models
    if safeguard:
        sizes_safe = [r.size_b for r in safeguard]
        detected_safe = [r.true_positives for r in safeguard]
        recall_safe = [r.recall * 100 for r in safeguard]

        ax.plot(sizes_safe, detected_safe, 'o-',
                color='#4ECDC4', linewidth=2, markersize=10,
                label=f'Safeguard GPT-OSS (specialized)', zorder=3)

    # Add reference line for total jailbreaks
    if results:
        total_jailbreaks = results[0].total_jailbreaks
        ax.axhline(y=total_jailbreaks, color='gray', linestyle='--',
                   linewidth=1, alpha=0.5, zorder=1,
                   label=f'Perfect detection ({total_jailbreaks} jailbreaks)')

    # Styling
    ax.set_xlabel('Model Size (Billions of Parameters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Jailbreaks Correctly Detected', fontsize=12, fontweight='bold')
    ax.set_title(f'Model Size vs Jailbreak Detection Accuracy\n'
                 f'(JailbreakHub Dataset, N={sample_size} samples)',
                 fontsize=14, fontweight='bold', pad=20)

    ax.set_xticks([20, 120])
    ax.set_xticklabels(['20B', '120B'])
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10, loc='lower right')

    # Add recall percentages as annotations
    for result in regular + safeguard:
        ax.annotate(f'{result.recall*100:.1f}%',
                   xy=(result.size_b, result.true_positives),
                   xytext=(0, 10), textcoords='offset points',
                   ha='center', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='gray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot: {output_path}")
    plt.close()


def create_comparison_table(results: Dict[int, List[ModelResult]], output_path: Path):
    """Create markdown table comparing all models"""

    with open(output_path, 'w') as f:
        f.write("# Model Comparison: Size vs Jailbreak Detection\n\n")
        f.write("## Summary of Results\n\n")

        for sample_size in sorted(results.keys()):
            if not results[sample_size]:
                continue

            f.write(f"### Dataset: {sample_size} samples\n\n")
            f.write("| Model | Type | Size | Detected | Total | Recall | Precision | F1 |\n")
            f.write("|-------|------|------|----------|-------|--------|-----------|----|\n")

            # Sort: regular first, then safeguard; within each, by size
            sorted_results = sorted(results[sample_size],
                                   key=lambda x: (x.model_type != "regular", x.size_b))

            for r in sorted_results:
                model_display = "Regular" if r.model_type == "regular" else "Safeguard"
                f.write(f"| {r.model_name} | {model_display} | {r.size_b}B | "
                       f"{r.true_positives}/{r.total_jailbreaks} | "
                       f"{r.total_jailbreaks} | "
                       f"{r.recall*100:.1f}% | "
                       f"{r.precision*100:.1f}% | "
                       f"{r.f1_score*100:.1f}% |\n")

            f.write("\n")

        # Add insights
        f.write("## Key Insights\n\n")

        # Size comparison (within same type)
        f.write("### 1. Impact of Model Size\n\n")
        for sample_size in sorted(results.keys()):
            if not results[sample_size]:
                continue

            for model_type in ["regular", "safeguard"]:
                models = [r for r in results[sample_size] if r.model_type == model_type]
                if len(models) >= 2:
                    models.sort(key=lambda x: x.size_b)
                    small = models[0]
                    large = models[-1]
                    improvement = (large.recall - small.recall) * 100

                    f.write(f"- **{model_type.title()} models ({sample_size} samples):** "
                           f"{small.size_b}B → {large.size_b}B = "
                           f"+{improvement:.1f}% improvement "
                           f"({small.recall*100:.1f}% → {large.recall*100:.1f}%)\n")

        f.write("\n### 2. Impact of Specialized Training\n\n")
        for sample_size in sorted(results.keys()):
            if not results[sample_size]:
                continue

            # Compare regular vs safeguard at each size
            for size in [20, 120]:
                regular = next((r for r in results[sample_size]
                              if r.model_type == "regular" and r.size_b == size), None)
                safeguard = next((r for r in results[sample_size]
                                if r.model_type == "safeguard" and r.size_b == size), None)

                if regular and safeguard:
                    improvement = (safeguard.recall - regular.recall) * 100
                    f.write(f"- **{size}B models ({sample_size} samples):** "
                           f"Regular → Safeguard = +{improvement:.1f}% improvement "
                           f"({regular.recall*100:.1f}% → {safeguard.recall*100:.1f}%)\n")

        f.write("\n### 3. Best Model\n\n")
        for sample_size in sorted(results.keys()):
            if not results[sample_size]:
                continue

            best = max(results[sample_size], key=lambda x: x.recall)
            f.write(f"- **{sample_size} samples:** {best.model_name} "
                   f"({best.recall*100:.1f}% recall, {best.f1_score*100:.1f}% F1)\n")

    print(f"✅ Saved comparison table: {output_path}")


def main():
    """Generate all visualization outputs"""

    print("=" * 80)
    print("Model Size vs Jailbreak Detection Visualization")
    print("=" * 80)

    # Setup paths
    results_dir = Path("experiments/jailbreak-evals/results")
    output_dir = Path("experiments/jailbreak-evals/plots")
    output_dir.mkdir(exist_ok=True)

    # Load results
    print("\nLoading evaluation results...")
    results = load_all_results(results_dir)

    # Generate plots
    print("\nGenerating plots...")

    if results[400]:
        print(f"\n📊 Creating 400-sample comparison plot...")
        create_comparison_plot(
            results[400],
            sample_size=400,
            output_path=output_dir / "model_comparison_400samples.png"
        )
    else:
        print("⚠️  No 400-sample results found yet")

    if results[5900]:
        print(f"\n📊 Creating 5900-sample comparison plot...")
        create_comparison_plot(
            results[5900],
            sample_size=5900,
            output_path=output_dir / "model_comparison_5900samples.png"
        )
    else:
        print("⚠️  No 5900-sample results found yet")

    # Generate comparison table
    print("\n📝 Creating comparison table...")
    create_comparison_table(results, output_dir / "model_comparison_table.md")

    print("\n" + "=" * 80)
    print("✅ Visualization complete!")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}/")
    print("- model_comparison_400samples.png")
    print("- model_comparison_5900samples.png")
    print("- model_comparison_table.md")


if __name__ == "__main__":
    main()
