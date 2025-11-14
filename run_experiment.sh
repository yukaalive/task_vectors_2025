#!/bin/bash

# Force use of GPU 1 (H100) instead of GPU 0 (V100)
export CUDA_VISIBLE_DEVICES=1

# Run the experiment
python scripts/experiments/main.py "$@"
