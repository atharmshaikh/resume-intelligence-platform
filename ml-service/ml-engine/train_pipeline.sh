#!/bin/bash
# train_pipeline.sh
# -----------------
# Linux Automation Script for Resume ML Pipeline.

# Ensure we are in the correct directory (ml-engine)
if [[ ! -d "ml_engine" ]]; then
    echo "Error: Please run this script from the ml-service/ml-engine/ directory."
    exit 1
fi

export PYTHONPATH=$PYTHONPATH:.
VENV_PATH="../../.venv/bin/python"

echo "────────────────────────────────────────────────────────────"
echo "   INDUSTRIAL RESUME ML PIPELINE - AUTOMATION (LINUX)       "
echo "────────────────────────────────────────────────────────────"

# 1. Environment Check
echo "[1/4] Validating Virtual Environment..."
if [ ! -f "$VENV_PATH" ]; then
    echo "❌ Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi
echo "      Found .venv/bin/python"

# 2. Syncing Processed Data
echo "[2/4] Syncing Processed Resumes (JSON Generation)..."
$VENV_PATH ml_engine/ml/inference/batch_processor.py > /dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: Batch processor failed. Check your resume files."
    exit 1
fi

# 3. Running Training
echo "[3/4] Starting Training Pipeline (JSON-First)..."
$VENV_PATH ml_engine/ml/pipelines/training.py
if [ $? -ne 0 ]; then
    echo "❌ Error: Training pipeline failed."
    exit 1
fi

# 4. Verification Pass
echo "[4/4] Post-Training Verification..."
$VENV_PATH ml_engine/ml/inference/batch_processor.py
echo ""
echo "✅ SUCCESS: Industrial Model deployed to ml_engine/ml/artifacts/"
echo "────────────────────────────────────────────────────────────"
