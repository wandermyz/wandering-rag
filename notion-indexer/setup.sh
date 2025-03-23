#!/bin/bash

# Exit on error
set -e

echo "Setting up notion-indexer environment..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Activate virtual environment
source .venv/bin/activate

# Generate locked requirements.txt from requirements.in
echo "Generating locked requirements.txt..."
uv pip compile requirements.in --output-file requirements.txt

# Install dependencies
echo "Installing dependencies..."
uv pip sync requirements.txt

echo "Setup complete! 🎉"
echo "To activate the virtual environment, run: source .venv/bin/activate" 