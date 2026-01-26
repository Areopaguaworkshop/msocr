#!/bin/bash

# Setup script for Manuscript OCR project

set -e

echo "🔧 Setting up Manuscript OCR Project..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install uv first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✅ uv found"

# Install dependencies
echo "📦 Installing dependencies..."
uv install

# Create project directories
echo "📁 Creating project directories..."
uv run python -c "
from msocr.utils import create_project_directories
dirs = create_project_directories('.')
print('✅ Created directories:')
for name, path in dirs.items():
    print(f'  {name}: {path}')
"

# Check if Kraken is available
echo "🔍 Checking Kraken installation..."
if uv run python -c "import kraken; print('✅ Kraken is available')" 2>/dev/null; then
    echo "✅ Kraken is properly installed"
else
    echo "⚠️  Kraken installation may need additional system dependencies"
    echo "   Please refer to: https://kraken.re/main/installation.html"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Add your manuscript images to data/images/"
echo "2. Run preprocessing: uv run msocr preprocess --input-dir data/images"
echo "3. Set up annotation: Check the notebooks in notebooks/"
echo "4. Train models: uv run marimo edit notebooks/sogdian_training.py"
echo ""
echo "📚 For detailed instructions, see README.md"