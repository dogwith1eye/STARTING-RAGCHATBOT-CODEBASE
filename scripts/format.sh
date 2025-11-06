#!/usr/bin/env bash

# Format code with isort and black
# Usage: ./scripts/format.sh [--check]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Running isort..."
if [ "$1" = "--check" ]; then
    uv run isort --check-only backend/ main.py
else
    uv run isort backend/ main.py
fi

echo ""
echo "Running black..."
if [ "$1" = "--check" ]; then
    uv run black --check backend/ main.py
else
    uv run black backend/ main.py
fi

echo ""
echo "✅ Formatting complete!"
