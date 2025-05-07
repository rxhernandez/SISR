#!/bin/bash

# Deactivate any active virtual environment
deactivate 2>/dev/null

# Remove the virtual environment directory
if [ -d "python_venvs/SISR" ]; then
    echo "Removing virtual environment at python_venvs/SISR"
    rm -rf python_venvs/SISR
fi

# Remove build artifacts
rm -rf python_venvs
rm -rf build dist *.egg-info
rm -rf src/*.egg-info
rm -rf src/SISR/__pycache__
rm -rf tests/__pycache__
rm -rf tests/*.txt

echo "Cleanup complete."
