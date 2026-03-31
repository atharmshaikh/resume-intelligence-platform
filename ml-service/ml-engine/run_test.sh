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
    echo "❌  .venv not found at: $PROJECT_ROOT/.venv"
    echo "    Run:  python -m venv .venv && .venv/bin/pip install -e ml-service/ml-engine"
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

# Clear bytecode cache to avoid stale-module issues
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Run the pipeline test with a clean explained output
"$VENV_PYTHON" tests/test_pipeline.py

EXIT_CODE=$?

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
