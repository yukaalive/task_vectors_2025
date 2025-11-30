"""
Test script for bidirectional averaged task vector experiment.

This script tests the new evaluate_bidirectional_averaged_task_vector function
without affecting existing code.
"""
import os
import sys

# Set CUDA device before any imports
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from scripts.experiments.main import run_bidirectional_averaged_experiment
from core.experiments_config import MODELS_TO_EVALUATE


def main():
    """
    Run bidirectional averaged task vector experiment for en_ja_easy and ja_en_easy.
    """
    # Test with llama_13B (or first available model)
    if len(sys.argv) >= 3:
        model_type = sys.argv[1]
        model_variant = sys.argv[2]
    else:
        # Default to first model in config
        model_type, model_variant = MODELS_TO_EVALUATE[0]

    print(f"Running bidirectional averaged experiment with {model_type}_{model_variant}")
    print("Tasks: translation_en_ja_easy + translation_ja_en_easy")
    print()

    # Run the experiment
    run_bidirectional_averaged_experiment(
        model_type=model_type,
        model_variant=model_variant,
        task1_name="translation_en_ja_easy",
        task2_name="translation_ja_en_easy",
        num_examples=5
    )

    print("\n✅ Experiment completed!")
    print(f"Results saved in outputs/results/main/bidirectional_avg/{model_type}_{model_variant}.pkl")


if __name__ == "__main__":
    main()
