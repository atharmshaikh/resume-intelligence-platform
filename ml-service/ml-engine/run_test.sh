#!/usr/bin/env bash
# =============================================================================
# run_test.sh  –  One-command test runner for ml-engine
#
# Usage (from any directory inside the project):
#   bash ml-service/ml-engine/run_test.sh
#
# Or from ml-engine/:
#   bash run_test.sh
# =============================================================================

set -euo pipefail

# ── Locate the project root (folder containing .venv) ────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

# ── Sanity checks ────────────────────────────────────────────────────────────
if [[ ! -f "$VENV_PYTHON" ]]; then
    echo "❌  CRITICAL ERROR: Virtual Environment not found at: $PROJECT_ROOT/.venv"
    echo "    Security requires execution isolated within a .venv context."
    echo "    Run:  python -m venv .venv && .venv/bin/pip install -e ml-service/ml-engine"
    exit 1
fi

# ── Require Python 3.12+ ─────────────────────────────────────────────────────
PYTHON_VERSION=$("$VENV_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ $(echo "$PYTHON_VERSION < 3.12" | bc -l 2>/dev/null || awk -v pv="$PYTHON_VERSION" 'BEGIN {print (pv < 3.12)}') -eq 1 ]]; then
    echo "❌  Security/Stability Error: Python $PYTHON_VERSION detected."
    echo "    This project strictly forces Python 3.12+ to prevent unknown bugs and vulnerabilities."
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🧪  Resume ML-Engine Test Suite"
echo "  Python : $VENV_PYTHON"
echo "  Engine : $SCRIPT_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd "$SCRIPT_DIR"

# Clear bytecode cache to avoid stale-module bugs (Industry Standard)
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# ── Anti Resource Killer & Overload Protection ───────────────────────────────
# Enforce a 3GB virtual memory limit to prevent memory leaks throwing OS errors
ulimit -v 3145728 2>/dev/null || true

# Run the pipeline test securely with a 5-minute timeout to prevent hanging/blocking
timeout 300 "$VENV_PYTHON" tests/test_pipeline.py

EXIT_CODE=$?

if [[ $EXIT_CODE -eq 124 ]]; then
    echo ""
    echo "❌  CRITICAL ALERT: Process killed by ANTI-BLOCK. Test execution exceeded 5-minute timeout threshold."
    exit 124
fi

if [[ $EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ✅  All tests passed"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ❌  Tests FAILED (exit code $EXIT_CODE)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi

exit $EXIT_CODE
