#!/bin/bash
set -e

ENV_NAME="sisr-env"
PYTHON_VERSION="3.11"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "conda could not be found. Please install Anaconda or Miniconda."
    exit 1
fi

# Create the environment if it doesn't exist
if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -y -n $ENV_NAME python=$PYTHON_VERSION
else
    echo "Conda environment '$ENV_NAME' already exists."
fi

echo "Activating conda environment '$ENV_NAME'..."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "Installing dependencies and the package..."
pip install --upgrade pip setuptools wheel
pip install .

echo "Running tests..."
python -m unittest discover -s tests -v

echo "Installation complete. To activate this environment later, run:"
echo "  conda activate $ENV_NAME"
