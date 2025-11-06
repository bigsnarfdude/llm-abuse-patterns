#!/bin/bash
#
# Full Dataset Evaluation - Nigel Server Deployment Script
# ---------------------------------------------------------
# Run complete JailbreakHub evaluation on nigel.birs.ca
#
# Usage:
#   ./run_on_nigel.sh [full|stratified|jailbreaks|test]
#
# Modes:
#   full        - Complete 15,140 dataset (~46 hours)
#   stratified  - 5,000 stratified sample (~15 hours) [DEFAULT]
#   jailbreaks  - 1,405 jailbreaks only (~4.3 hours)
#   test        - 100 prompt test run (~18 minutes)

set -e  # Exit on error

# Configuration
MODE=${1:-stratified}
MODEL="safeguard"
EXPERIMENT_DIR="experiments/full-dataset-eval"
LOG_DIR="logs"
CHECKPOINT_DIR="checkpoints"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "Full Dataset Evaluation - Nigel Deployment"
echo "========================================="
echo ""
echo "Mode: $MODE"
echo "Model: $MODEL"
echo ""

# Create directories
mkdir -p $LOG_DIR
mkdir -p $CHECKPOINT_DIR

# Check if we're on nigel
if [[ $(hostname) != *"nigel"* ]]; then
    echo -e "${RED}⚠️  WARNING: Not running on nigel.birs.ca${NC}"
    echo "This script is designed for nigel server."
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Ollama
echo "Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}❌ Ollama not found. Please install Ollama first.${NC}"
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo -e "${YELLOW}⚠️  Ollama not running. Starting Ollama...${NC}"
    ollama serve &
    sleep 5
fi

# Check if model is available
echo "Checking for gpt-oss-safeguard model..."
if ! ollama list | grep -q "gpt-oss-safeguard"; then
    echo -e "${YELLOW}⚠️  Model not found. Pulling gpt-oss-safeguard:latest...${NC}"
    ollama pull gpt-oss-safeguard:latest
fi

echo -e "${GREEN}✅ Ollama and model ready${NC}"
echo ""

# Check Python dependencies
echo "Checking Python dependencies..."
if ! python3 -c "import datasets" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  datasets library not found. Installing...${NC}"
    pip install datasets
fi

echo -e "${GREEN}✅ Dependencies ready${NC}"
echo ""

# Build command based on mode
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$LOG_DIR/${MODE}_${MODEL}_${TIMESTAMP}.txt"
LOG_FILE="$LOG_DIR/run_${MODE}_${TIMESTAMP}.log"

case $MODE in
    full)
        echo "========================================="
        echo "FULL DATASET EVALUATION"
        echo "========================================="
        echo "Dataset: 15,140 prompts"
        echo "Expected time: ~46 hours"
        echo "Checkpoints: Every 100 prompts"
        echo ""

        PYTHON_CMD="python3 08_full_dataset_llm_eval.py \
            --model $MODEL \
            --checkpoint \
            --checkpoint-dir $CHECKPOINT_DIR \
            --batch-size 100 \
            --output $OUTPUT_FILE"
        ;;

    stratified)
        echo "========================================="
        echo "STRATIFIED SAMPLE EVALUATION"
        echo "========================================="
        echo "Dataset: 5,000 prompts (9.3% jailbreaks)"
        echo "Expected time: ~15 hours"
        echo "Checkpoints: Every 100 prompts"
        echo ""

        PYTHON_CMD="python3 08_full_dataset_llm_eval.py \
            --model $MODEL \
            --sample-size 5000 \
            --stratified \
            --checkpoint \
            --checkpoint-dir $CHECKPOINT_DIR \
            --batch-size 100 \
            --output $OUTPUT_FILE"
        ;;

    jailbreaks)
        echo "========================================="
        echo "JAILBREAKS-ONLY EVALUATION"
        echo "========================================="
        echo "Dataset: 1,405 jailbreak prompts"
        echo "Expected time: ~4.3 hours"
        echo "Checkpoints: Every 100 prompts"
        echo ""

        PYTHON_CMD="python3 08_full_dataset_llm_eval.py \
            --model $MODEL \
            --jailbreaks-only \
            --checkpoint \
            --checkpoint-dir $CHECKPOINT_DIR \
            --batch-size 100 \
            --output $OUTPUT_FILE"
        ;;

    test)
        echo "========================================="
        echo "TEST RUN (100 prompts)"
        echo "========================================="
        echo "Dataset: 100 prompts (balanced)"
        echo "Expected time: ~18 minutes"
        echo ""

        PYTHON_CMD="python3 08_full_dataset_llm_eval.py \
            --model $MODEL \
            --sample-size 100 \
            --output $OUTPUT_FILE"
        ;;

    *)
        echo -e "${RED}❌ Invalid mode: $MODE${NC}"
        echo "Valid modes: full, stratified, jailbreaks, test"
        exit 1
        ;;
esac

echo "Output: $OUTPUT_FILE"
echo "Logs: $LOG_FILE"
echo "Checkpoints: $CHECKPOINT_DIR"
echo ""

# Confirm before running
if [[ $MODE != "test" ]]; then
    echo -e "${YELLOW}⚠️  This will take several hours to complete.${NC}"
    read -p "Start evaluation now? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Evaluation cancelled."
        exit 0
    fi
fi

echo ""
echo "========================================="
echo "STARTING EVALUATION"
echo "========================================="
echo ""
echo "Command: $PYTHON_CMD"
echo ""
echo "Press Ctrl+C to stop (progress will be saved in checkpoints)"
echo ""

# Change to experiment directory
cd $EXPERIMENT_DIR

# Run evaluation (pipe to tee for both file and stdout)
$PYTHON_CMD 2>&1 | tee $LOG_FILE

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ EVALUATION COMPLETE${NC}"
    echo ""
    echo "Results saved to: $OUTPUT_FILE"
    echo "Logs saved to: $LOG_FILE"
    echo ""

    # Show summary if available
    if [ -f "$OUTPUT_FILE" ]; then
        echo "========================================="
        echo "RESULTS SUMMARY"
        echo "========================================="
        tail -30 "$LOG_FILE"
    fi
else
    echo ""
    echo -e "${RED}❌ EVALUATION FAILED${NC}"
    echo ""
    echo "Check logs: $LOG_FILE"
    echo "Resume with: python3 08_full_dataset_llm_eval.py --resume $CHECKPOINT_DIR/latest.json"
    exit 1
fi
