#!/usr/bin/env bash

# Run all code quality checks
# Usage: ./scripts/quality.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Running Code Quality Checks"
echo "=========================================="
echo ""

# Format check
echo "1. Checking code formatting..."
echo "------------------------------------------"
./scripts/format.sh --check

echo ""
echo "2. Running linters..."
echo "------------------------------------------"
./scripts/lint.sh

echo ""
echo "3. Running tests..."
echo "------------------------------------------"
cd backend && uv run pytest tests/ -v

echo ""
echo "=========================================="
echo "✅ All quality checks passed!"
echo "=========================================="
