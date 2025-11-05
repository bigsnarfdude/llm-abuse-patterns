#!/usr/bin/env python3
"""Quick debug test - just 2 prompts with verbose output"""

import sys
sys.path.insert(0, 'src')

from datasets import load_dataset
from llm_abuse_patterns.safeguard import SafeguardDetector
import time

print("Loading dataset...")
dataset = load_dataset('walledai/JailbreakHub', split='train')
print(f"✅ Loaded {len(dataset)} prompts\n")

# Get one jailbreak and one benign
jailbreak = [x for x in dataset if x['jailbreak']][0]
benign = [x for x in dataset if not x['jailbreak']][0]

print("=" * 80)
print("Test 1: Jailbreak Prompt")
print("=" * 80)
print(f"Prompt (first 200 chars): {jailbreak['prompt'][:200]}...")
print(f"Ground truth: JAILBREAK\n")

print("Testing SafeguardDetector...")
detector = SafeguardDetector(model="gpt-oss:20b", reasoning_effort="medium")
start = time.time()
result = detector.detect(jailbreak['prompt'])
elapsed = time.time() - start
print(f"Result: {'🚨 JAILBREAK' if result.is_jailbreak else '✅ SAFE'}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Latency: {elapsed:.1f}s")
print(f"Reasoning: {result.reasoning}\n")

print("=" * 80)
print("Test 2: Benign Prompt")
print("=" * 80)
print(f"Prompt: {benign['prompt'][:200]}...")
print(f"Ground truth: BENIGN\n")

print("Testing SafeguardDetector...")
start = time.time()
result = detector.detect(benign['prompt'])
elapsed = time.time() - start
print(f"Result: {'🚨 JAILBREAK' if result.is_jailbreak else '✅ SAFE'}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Latency: {elapsed:.1f}s")
print(f"Reasoning: {result.reasoning}\n")

print("✅ Quick test complete!")
