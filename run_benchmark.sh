#!/bin/bash

# Exit on error
set -e

# Source conda
source "${HOME}/conda/etc/profile.d/conda.sh"

# Activate conda environment
conda activate lemon


python benchmark_estimator_generic.py "$@"
