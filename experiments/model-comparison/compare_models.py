#!/usr/bin/env python3
"""
Compare Results: gpt-oss:20b vs gpt-oss-safeguard:20b
------------------------------------------------------
Analyzes evaluation results from both models to show performance differences.

Usage:
    python compare_models.py jailbreakhub_400_full.txt jailbreakhub_400_safeguard.txt
"""

import sys
import re
from typing import Dict, Optional

def parse_results(filename: str) -> Optional[Dict]:
    """Parse evaluation results from output file"""
    try:
        with open(filename, 'r') as f:
            content = f.read()

        # Extract metrics table
        results = {}

        # Find the results table
        table_match = re.search(
            r'Method\s+Precision\s+Recall\s+F1\s+Accuracy\s+Latency.*?\n-+\n(.*?)(?:\n\n|====)',
            content, re.DOTALL
        )

        if not table_match:
            print(f"⚠️  Could not find results table in {filename}")
            return None

        table_lines = table_match.group(1).strip().split('\n')

        for line in table_lines:
            # Parse lines like: "Heuristic    96.0% 24.0% 38.4% 61.5% 0.2ms"
            parts = line.split()
            if len(parts) >= 5:
                method = parts[0]
                if 'Layered' in line:
                    method = 'Layered'
                elif 'Real-LLM' in line:
                    method = 'Real-LLM'
                elif 'Heuristic' in line:
                    method = 'Heuristic'

                # Extract percentages and latency
                precision = float(parts[-5].rstrip('%'))
                recall = float(parts[-4].rstrip('%'))
                f1 = float(parts[-3].rstrip('%'))
                accuracy = float(parts[-2].rstrip('%'))
                latency_str = parts[-1]

                results[method] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'accuracy': accuracy,
                    'latency': latency_str
                }

        # Extract confusion matrix details
        cm_match = re.search(
            r'DETAILED CONFUSION MATRIX.*?Real-LLM:\s+True Positives.*?:\s+(\d+).*?False Negatives.*?:\s+(\d+).*?True Negatives.*?:\s+(\d+).*?False Positives.*?:\s+(\d+)',
            content, re.DOTALL
        )

        if cm_match:
            results['confusion_matrix'] = {
                'tp': int(cm_match.group(1)),
                'fn': int(cm_match.group(2)),
                'tn': int(cm_match.group(3)),
                'fp': int(cm_match.group(4))
            }

        return results

    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        return None
    except Exception as e:
        print(f"❌ Error parsing {filename}: {e}")
        return None


def compare_results(results1: Dict, results2: Dict, label1: str, label2: str):
    """Print comparison of two result sets"""

    print("=" * 100)
    print("MODEL COMPARISON: gpt-oss:20b vs gpt-oss-safeguard:20b")
    print("=" * 100)

    print(f"\n📊 Performance Comparison:\n")

    # Header
    print(f"{'Method':<20} {'Metric':<12} {label1:<25} {label2:<25} {'Difference':<15}")
    print("-" * 100)

    methods = ['Heuristic', 'Real-LLM', 'Layered']
    metrics = ['precision', 'recall', 'f1', 'accuracy']

    for method in methods:
        if method not in results1 or method not in results2:
            continue

        print(f"\n{method}:")
        for metric in metrics:
            val1 = results1[method][metric]
            val2 = results2[method][metric]
            diff = val2 - val1
            diff_str = f"{diff:+.1f}%" if diff != 0 else "same"

            # Color code the difference
            if diff > 0:
                diff_str = f"✅ {diff_str}"
            elif diff < 0:
                diff_str = f"❌ {diff_str}"

            print(f"  {metric.capitalize():<18} {val1:>6.1f}%               {val2:>6.1f}%               {diff_str}")

        # Latency comparison
        lat1 = results1[method]['latency']
        lat2 = results2[method]['latency']
        print(f"  {'Latency':<18} {lat1:>10}            {lat2:>10}")

    print("\n" + "=" * 100)
    print("KEY FINDINGS:")
    print("=" * 100)

    # Calculate overall improvement
    llm_f1_diff = results2['Real-LLM']['f1'] - results1['Real-LLM']['f1']
    layered_f1_diff = results2['Layered']['f1'] - results1['Layered']['f1']

    print(f"\n🎯 Real-LLM F1 Score: {llm_f1_diff:+.1f}% {'improvement' if llm_f1_diff > 0 else 'decline'}")
    print(f"🏆 Layered Defense F1 Score: {layered_f1_diff:+.1f}% {'improvement' if layered_f1_diff > 0 else 'decline'}")

    # Recall comparison (most important for security)
    llm_recall_diff = results2['Real-LLM']['recall'] - results1['Real-LLM']['recall']
    layered_recall_diff = results2['Layered']['recall'] - results1['Layered']['recall']

    print(f"\n🔍 Real-LLM Recall: {llm_recall_diff:+.1f}% {'(catching more jailbreaks!)' if llm_recall_diff > 0 else '(catching fewer)'}")
    print(f"🛡️  Layered Recall: {layered_recall_diff:+.1f}% {'(catching more jailbreaks!)' if layered_recall_diff > 0 else '(catching fewer)'}")

    print("\n" + "=" * 100)


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_models.py <results_file_1> <results_file_2>")
        print("Example: python compare_models.py jailbreakhub_400_full.txt jailbreakhub_400_safeguard.txt")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    print(f"\n📂 Loading results from:")
    print(f"   Model 1: {file1}")
    print(f"   Model 2: {file2}\n")

    results1 = parse_results(file1)
    results2 = parse_results(file2)

    if not results1 or not results2:
        print("\n❌ Could not load both result files. Please check file paths and formats.")
        sys.exit(1)

    compare_results(
        results1, results2,
        "gpt-oss:20b (original)",
        "gpt-oss-safeguard:20b (official)"
    )

    print("\n✅ Comparison complete!\n")


if __name__ == "__main__":
    main()
