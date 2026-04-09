#!/bin/bash

# =============================================================================
# Resume ML Engine - Batch Run Script (Linux/macOS)
# =============================================================================

# Define venv path relative to this script
VENV_DIR="../../.venv"

<<<<<<< HEAD
=======
# 0. Ensure we are in the correct directory
cd "$(dirname "$0")"

>>>>>>> feature/optimization-and-refactor
# 1. Check if venv exists, create if missing
if [ ! -d "$VENV_DIR" ]; then
    echo "🏗️  Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    
    echo "📦  Installing dependencies from pyproject.toml..."
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -e .
fi

# 2. Activate and Run
echo "🚀  Running Resume ML Batch Processor..."
"$VENV_DIR/bin/python" -m ml_engine.ml.inference.batch_processor "$@"
