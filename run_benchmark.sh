#!/bin/bash

# Exit on error
set -e

# Vendor-specific setup
if [ "$1" = "NVIDIA" ]; then
    echo "Detected NVIDIA vendor - sourcing conda..."
    source "${HOME}/conda/etc/profile.d/conda.sh"
fi

if [ "$1" = "APPLE" ]; then
    echo "Detected APPLE vendor - ensuring mlx-lm is installed..."
    conda run -n lemon pip install mlx-lm --quiet
fi

conda run -n lemon python benchmark_estimator_generic.py "$@"
