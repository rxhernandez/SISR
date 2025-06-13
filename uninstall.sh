#!/bin/bash
set -e

ENV_NAME="sisr-env"

if ! command -v conda &> /dev/null; then
    echo "conda could not be found. Please install Anaconda or Miniconda."
    exit 1
fi

echo "Deactivating any active conda environment..."
conda deactivate || true

if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Removing conda environment '$ENV_NAME'..."
    conda remove -y -n $ENV_NAME --all
else
    echo "Conda environment '$ENV_NAME' does not exist."
fi

echo "Cleanup complete."
