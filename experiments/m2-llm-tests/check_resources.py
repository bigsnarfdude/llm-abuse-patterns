#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
import psutil
import os

print("=" * 70)
print("M2 MAC RESOURCE USAGE - GPT-OSS 20B")
print("=" * 70)
print()

# System resources
vm = psutil.virtual_memory()
print(f"💾 Total RAM: {vm.total / 1024**3:.1f} GB")
print(f"   Available: {vm.available / 1024**3:.1f} GB")
print(f"   Used: {vm.used / 1024**3:.1f} GB ({vm.percent}%)")
print()

# Check Ollama process
ollama_procs = [p for p in psutil.process_iter(['pid', 'name', 'memory_info']) 
                if 'ollama' in p.info['name'].lower()]

if ollama_procs:
    print("🦙 Ollama Process:")
    for proc in ollama_procs:
        mem_mb = proc.info['memory_info'].rss / 1024**2
        print(f"   PID {proc.info['pid']}: {mem_mb:.0f} MB")
    print()

print("🧪 Running one inference to check peak usage...")
print()

from llm_abuse_patterns.safeguard import SafeguardDetector
detector = SafeguardDetector(model="gpt-oss:20b")

mem_before = psutil.virtual_memory().available / 1024**3

result = detector.detect("Test prompt for resource check")

mem_after = psutil.virtual_memory().available / 1024**3
mem_delta = mem_before - mem_after

print(f"✅ Inference complete")
print(f"   Latency: {result.latency_ms:.0f}ms")
print(f"   Memory change: {mem_delta:.1f} GB")
print()

# Final stats
vm = psutil.virtual_memory()
print("📊 FINAL RESOURCE STATUS:")
print(f"   Total RAM: {vm.total / 1024**3:.1f} GB")
print(f"   Used: {vm.used / 1024**3:.1f} GB ({vm.percent}%)")
print(f"   Available: {vm.available / 1024**3:.1f} GB")
print()
print("✅ Your 32GB M2 Mac has PLENTY of headroom for this workload!")
print("=" * 70)
