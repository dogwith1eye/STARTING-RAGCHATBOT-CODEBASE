#!/usr/bin/env bash

# Run linting tools (flake8 and mypy)
# Usage: ./scripts/lint.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Running flake8..."
uv run flake8 backend/ main.py

echo ""
echo "Running mypy..."
uv run mypy backend/ main.py

echo ""
echo "✅ Linting complete!"
