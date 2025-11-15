#!/usr/bin/env python3
"""
ToxicChat Evaluation - Dedicated Classifier vs Safeguard
========================================================

Testing OpenAI's claim from technical report:
"Classifiers trained on tens of thousands of high-quality labeled samples
can still perform better at classifying content than gpt-oss-safeguard
does when reasoning directly from the policy."

Experiment:
1. Train simple classifier on ToxicChat training data (5,082 samples)
2. Compare with gpt-oss-safeguard reasoning from policy
3. Measure accuracy, speed, and cost trade-offs

Models to test:
- Logistic Regression (baseline)
- Random Forest
- Small BERT (DistilBERT)
- gpt-oss-safeguard (policy-based)

Expected: Dedicated classifiers beat safeguard on F1 but lack flexibility
"""

import time
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report

def train_logistic_regression(X_train, y_train):
    """Train logistic regression classifier"""
    print("Training Logistic Regression...")
    start = time.time()

    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)

    train_time = time.time() - start
    print(f"  Training time: {train_time:.1f}s")

    return clf

def train_random_forest(X_train, y_train):
    """Train random forest classifier"""
    print("Training Random Forest...")
    start = time.time()

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    train_time = time.time() - start
    print(f"  Training time: {train_time:.1f}s")

    return clf

def evaluate_classifier(clf, X_test, y_test, name):
    """Evaluate classifier and return metrics"""
    print(f"\nEvaluating {name}...")

    # Predict
    start = time.time()
    y_pred = clf.predict(X_test)
    inference_time = (time.time() - start) / len(y_test) * 1000  # ms per sample

    # Metrics
    f1 = f1_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred) * 100
    recall = recall_score(y_test, y_pred) * 100

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        'name': name,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'latency_ms': inference_time
    }

def main():
    print("=" * 80)
    print("DEDICATED CLASSIFIER vs SAFEGUARD COMPARISON")
    print("=" * 80)
    print()
    print("Testing OpenAI's claim:")
    print('"Classifiers trained on tens of thousands of high-quality labeled')
    print('samples can still perform better than gpt-oss-safeguard"')
    print()

    # Load dataset
    print("Loading ToxicChat dataset...")
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124")

    train_data = ds['train']
    test_data = ds['test']

    print(f"Training set: {len(train_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    print()

    # Prepare data
    print("Extracting features (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

    X_train = vectorizer.fit_transform([x['user_input'] for x in train_data])
    y_train = np.array([x['toxicity'] for x in train_data])

    X_test = vectorizer.transform([x['user_input'] for x in test_data])
    y_test = np.array([x['toxicity'] for x in test_data])

    print(f"  Features: {X_train.shape[1]}")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print()

    # Train classifiers
    print("=" * 80)
    print("TRAINING DEDICATED CLASSIFIERS")
    print("=" * 80)
    print()

    lr_clf = train_logistic_regression(X_train, y_train)
    rf_clf = train_random_forest(X_train, y_train)

    print()

    # Evaluate
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    results = []

    # Logistic Regression
    lr_results = evaluate_classifier(lr_clf, X_test, y_test, "Logistic Regression")
    results.append(lr_results)

    # Random Forest
    rf_results = evaluate_classifier(rf_clf, X_test, y_test, "Random Forest")
    results.append(rf_results)

    # Add safeguard results from previous experiments
    safeguard_original = {
        'name': 'Safeguard (Original Policy)',
        'f1': 70.1,
        'precision': 84.4,
        'recall': 59.9,
        'tp': 217,
        'fp': 40,
        'tn': 4681,
        'fn': 145,
        'latency_ms': 700  # 0.7s
    }
    results.append(safeguard_original)

    print()
    print("=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print()

    print(f"{'Model':<30} {'F1':>8} {'Recall':>8} {'Precision':>10} {'Latency':>10}")
    print("-" * 80)

    for r in results:
        print(f"{r['name']:<30} {r['f1']:>7.1f}% {r['recall']:>7.1f}% {r['precision']:>9.1f}% {r['latency_ms']:>8.0f}ms")

    print()
    print("=" * 80)
    print("DETAILED CONFUSION MATRICES")
    print("=" * 80)
    print()

    for r in results:
        print(f"{r['name']}:")
        print(f"  TP: {r['tp']:>4} | FP: {r['fp']:>4}")
        print(f"  FN: {r['fn']:>4} | TN: {r['tn']:>4}")
        print()

    # Analysis
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    best_f1 = max(results, key=lambda x: x['f1'])
    fastest = min(results, key=lambda x: x['latency_ms'])
    best_precision = max(results, key=lambda x: x['precision'])
    best_recall = max(results, key=lambda x: x['recall'])

    print(f"🏆 Best F1: {best_f1['name']} ({best_f1['f1']:.1f}%)")
    print(f"⚡ Fastest: {fastest['name']} ({fastest['latency_ms']:.0f}ms)")
    print(f"🎯 Best Precision: {best_precision['name']} ({best_precision['precision']:.1f}%)")
    print(f"📊 Best Recall: {best_recall['name']} ({best_recall['recall']:.1f}%)")
    print()

    # Validate OpenAI's claim
    dedicated_best = max([r for r in results if 'Safeguard' not in r['name']], key=lambda x: x['f1'])
    safeguard_best = max([r for r in results if 'Safeguard' in r['name']], key=lambda x: x['f1'])

    print("OpenAI's Claim Validation:")
    print(f"  Best dedicated classifier: {dedicated_best['name']} ({dedicated_best['f1']:.1f}% F1)")
    print(f"  Safeguard (policy-based):  {safeguard_best['name']} ({safeguard_best['f1']:.1f}% F1)")
    print()

    if dedicated_best['f1'] > safeguard_best['f1']:
        gap = dedicated_best['f1'] - safeguard_best['f1']
        print(f"✅ CLAIM VALIDATED: Dedicated classifier beats safeguard by {gap:.1f}%")
    else:
        gap = safeguard_best['f1'] - dedicated_best['f1']
        print(f"❌ CLAIM REJECTED: Safeguard beats dedicated classifier by {gap:.1f}%")

    print()
    print("=" * 80)
    print("TRADE-OFF ANALYSIS")
    print("=" * 80)
    print()

    print("Dedicated Classifiers (Logistic Regression, Random Forest):")
    print("  ✅ Pros:")
    print("    - Very fast inference (<1ms)")
    print("    - Low cost (no LLM calls)")
    print("    - Potentially higher F1 (if well-trained)")
    print("    - Predictable latency")
    print("  ❌ Cons:")
    print("    - Requires large labeled dataset")
    print("    - No reasoning/explainability")
    print("    - Fixed to training data distribution")
    print("    - Can't adapt to new policies without retraining")
    print("    - Domain-specific (can't generalize)")
    print()

    print("Safeguard (Policy-based Reasoning):")
    print("  ✅ Pros:")
    print("    - No training data required")
    print("    - Explainable reasoning (CoT)")
    print("    - Adaptable policies (change rules instantly)")
    print("    - Generalizes across domains")
    print("    - Understands context and nuance")
    print("  ❌ Cons:")
    print("    - Slower inference (~700ms)")
    print("    - Higher cost (LLM inference)")
    print("    - May have lower F1 than specialized models")
    print("    - Variable latency")
    print()

    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    print("When to use Dedicated Classifiers:")
    print("  1. Fixed, well-defined task (e.g., spam detection)")
    print("  2. Large labeled dataset available (10K+ samples)")
    print("  3. Low latency critical (<10ms)")
    print("  4. Cost is primary concern")
    print("  5. No need for explainability")
    print()

    print("When to use Safeguard (Policy-based):")
    print("  1. Evolving policies (rules change frequently)")
    print("  2. Limited labeled data")
    print("  3. Need explainability (why was content flagged?)")
    print("  4. Multi-domain (same model for different tasks)")
    print("  5. Latency acceptable (100-1000ms)")
    print()

    print("Hybrid Approach (Best of Both):")
    print("  1. Use dedicated classifier as fast pre-filter")
    print("  2. Send borderline cases to safeguard for reasoning")
    print("  3. Achieve both speed and accuracy")
    print()

    print("=" * 80)
    print("COST ANALYSIS (Rough Estimates)")
    print("=" * 80)
    print()

    # Estimate costs
    samples_per_day = 1_000_000

    lr_cost = samples_per_day * 0.0000001  # $0.0001 per 1K inferences
    safeguard_cost = samples_per_day * 0.005 / 1000  # $0.005 per 1K tokens

    print(f"Processing {samples_per_day:,} samples/day:")
    print(f"  Logistic Regression: ${lr_cost:.2f}/day")
    print(f"  Safeguard (20B):     ${safeguard_cost:.2f}/day")
    print(f"  Cost ratio:          {safeguard_cost/lr_cost:.0f}x more expensive")
    print()

    # Hybrid approach
    dedicated_filter_rate = 0.7  # 70% filtered by dedicated
    hybrid_cost = (samples_per_day * 0.7 * 0.0000001) + (samples_per_day * 0.3 * 0.005 / 1000)

    print(f"Hybrid Approach (70% dedicated, 30% safeguard):")
    print(f"  Cost: ${hybrid_cost:.2f}/day")
    print(f"  Savings: {(1 - hybrid_cost/safeguard_cost)*100:.0f}% vs pure safeguard")
    print()

    print("=" * 80)

if __name__ == "__main__":
    main()
