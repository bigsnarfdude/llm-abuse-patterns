# Quick test - just 10 prompts
import sys
sys.path.insert(0, 'src')
import jailbreakbench as jbb
from llm_abuse_patterns.safeguard import SafeguardDetector

print("Loading dataset...")
dataset = jbb.read_dataset()
print(f"✅ Loaded {len(dataset.behaviors)} behaviors")
print("\nTesting first 3 prompts with Real-LLM...")

detector = SafeguardDetector(model="gpt-oss:20b")

for i in range(3):
    goal = dataset.goals[i]
    category = dataset.categories[i]
    print(f"\n[{i+1}/3] Category: {category}")
    print(f"Goal: {goal[:80]}...")
    
    result = detector.detect(goal)
    print(f"Result: {'🚨 JAILBREAK' if result.is_jailbreak else '✅ SAFE'} (confidence: {result.confidence:.2%}, latency: {result.latency_ms/1000:.1f}s)")

print("\n✅ Quick test complete!")
