#!/bin/bash

# Development setup script for DINOv3 custom training project
# This script sets up the development environment with all necessary tools

set -e  # Exit on any error

echo "🚀 Setting up DINOv3 development environment..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if uv is available, install if not
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv (fast Python package installer)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Install dependencies
echo "📦 Installing Python dependencies with uv..."
uv pip install -r requirements.txt

# Install pre-commit hooks
echo "🔗 Setting up pre-commit hooks..."
pre-commit install

# Run pre-commit on all files to ensure everything is properly formatted
echo "🎨 Running initial code formatting..."
pre-commit run --all-files || true

# Setup done
echo "✅ Development environment setup complete!"
echo ""
echo "📋 Available development commands:"
echo "  - uv pip install <package>    : Install packages with uv"
echo "  - pre-commit run --all-files  : Run all pre-commit hooks"
echo "  - black .                     : Format code with Black"
echo "  - isort .                     : Sort imports"
echo "  - flake8 .                    : Run linting"
echo "  - mypy src/                   : Run type checking"
echo "  - pytest                     : Run tests"
echo ""
echo "🔧 Pre-commit hooks are now installed and will run automatically on git commits."
