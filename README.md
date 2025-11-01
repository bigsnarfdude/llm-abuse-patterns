# LLM Abuse & Safety Overwatch
Problem Statement:
- The potential harm is emerging or evolving, and policies need to adapt quickly.
- The domain is highly nuanced and difficult for smaller classifiers to handle.
- Developers don’t have enough samples to train a high-quality classifier for each risk on their platform.
- Latency is less important than producing high-quality, explainable labels.





Project Goal: To build and experiment with a comprehensive "overwatch" system for detecting and mitigating Large Language Model (LLM) abuse.

This project moves beyond simple prompt-rejection. It focuses on holistic, real-time traffic analysis and behavioral reasoning to identify malicious intent, abusive potential, and system backdoors. The core idea is that safety requires significant compute to actively monitor and reason about all interactions, much like a security overwatch.

## 🔬 Core Experimental Areas
This repository is organized around three main detection experiments:

1. Traffic-Based Intent & Misuse Detection
This experiment focuses on analyzing "all traffic" (prompts, responses, and API calls) to find patterns of abuse.

Methodology:

Dynamic Analysis: Simulates user misuse (e.g., malware generation, phishing, data exfiltration) to test model responses.

Behavioral Aggregation: Instead of just flagging single prompts, this module aggregates features at a user or relationship level. It reasons about behavior over time, such as repeated attempts to bypass filters or escalating toxic language.

Toxicity & Emotion Scoring: Uses smaller, faster BERT-based models to score all traffic for toxicity, threats, emotion, and sentiment, feeding this data into the behavioral model.

Repo Contains:

Scripts for dynamic misuse simulation.

Models for high-speed toxicity/emotion classification.

Experiments in aggregating user behavior features over time (e.g., user_history_analysis.py).

2. Static "Overwatch" Analysis (LLM App Security)
This experiment analyzes LLM-integrated applications before they are deployed to find "abusive potential."

Methodology:

Static Scanning: Scans all application components (instructions, knowledge files, action schemas) for risks.

Risk Detection: Identifies:

Malicious Intent: Detects toxic content or deceptive instructions.

Data Over-collection: Flags apps that request excessive permissions or PII.

Malicious Domains: Checks all embedded links and API endpoints against threat databases.

Repo Contains:

A static analyzer for LLM app manifests and knowledge files (static_analyzer/).

Datasets of flagged malicious instructions and schemas.

3. LLM as a Reasoning Layer (Hybrid Classification)
This experiment tests the trade-offs of using an LLM as the core "reasoning" engine for safety, balancing cost and accuracy.

Methodology:

Experiment 1: Simple Classifiers: Use computationally cheap models (e.g., Logistic Regression, Random Forest) on TF-IDF features for high-speed, basic filtering.

Experiment 2: LLM Classifier: Use a more powerful (but expensive) model like GPT-4 or a fine-tuned Llama 3 to classify nuanced or complex abuse prompts.

Hybrid Model: A "triage" system. Fast, cheap models handle 90% of traffic. Suspicious or complex interactions are escalated to the expensive LLM for deep reasoning.

Repo Contains:

Jupyter notebooks comparing the performance (F1, precision, recall, latency) and cost of all three methods.

Scripts for implementing the hybrid "triage" pipeline (hybrid_triage/).
