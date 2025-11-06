# Code Quality Tools

This directory contains scripts for maintaining code quality in the RAG Chatbot project.

## Available Scripts

### `format.sh` - Code Formatting

Automatically formats code using `isort` and `black`.

**Usage:**
```bash
# Format all Python files
./scripts/format.sh

# Check formatting without making changes
./scripts/format.sh --check
```

**What it does:**
- Sorts imports using `isort` (compatible with black)
- Formats code using `black` (88 character line length)

### `lint.sh` - Code Linting

Runs static analysis tools to check code quality.

**Usage:**
```bash
./scripts/lint.sh
```

**What it does:**
- Runs `flake8` for style checking
- Runs `mypy` for type checking

### `quality.sh` - Full Quality Check

Runs all quality checks including formatting, linting, and tests.

**Usage:**
```bash
./scripts/quality.sh
```

**What it does:**
1. Checks code formatting (isort + black)
2. Runs linters (flake8 + mypy)
3. Runs test suite (pytest)

## Configuration Files

### `.flake8`
Configures flake8 linting rules:
- Max line length: 100 characters
- Ignores: E203, W503, E501
- Per-file ignores for test files

### `pyproject.toml`
Contains configuration for:
- **black**: Code formatter settings
- **isort**: Import sorting settings
- **mypy**: Type checking settings (configured to be permissive)

## Installed Tools

All tools are installed as dev dependencies:

```bash
# Install/update dependencies
uv sync
```

**Tools included:**
- `black` (v25.1.0+): Opinionated code formatter
- `isort` (v5.13.0+): Import statement organizer
- `flake8` (v7.1.0+): Style guide enforcement
- `mypy` (v1.15.0+): Static type checker

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Format your code
./scripts/format.sh

# 3. Check for issues
./scripts/lint.sh

# 4. Run full quality check
./scripts/quality.sh
```

## Pre-commit Workflow

Before committing code, run:

```bash
./scripts/format.sh       # Auto-format code
./scripts/quality.sh      # Verify everything passes
```

## CI/CD Integration

These scripts can be integrated into CI/CD pipelines:

```bash
# In your CI pipeline
./scripts/format.sh --check  # Fail if code isn't formatted
./scripts/lint.sh            # Fail on linting errors
./scripts/quality.sh         # Full check including tests
```

## Notes

- The mypy configuration is currently permissive to avoid blocking development
- To make type checking stricter, edit `[tool.mypy]` in `pyproject.toml`
- Test files have relaxed linting rules (F401, F841, E402 ignored)
