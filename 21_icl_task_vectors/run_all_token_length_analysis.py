"""
experiments_config.pyに定義されているすべてのモデルとタスクでトークン長分析を実行するスクリプト
"""

from dotenv import load_dotenv
load_dotenv(".env")

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.experiments_config import MODELS_TO_EVALUATE, TASKS_TO_EVALUATE
from scripts.experiments.token_length_analysis import run_token_length_analysis


def main():
    print("=" * 80)
    print("Token Length Analysis - Running All Configured Models and Tasks")
    print("=" * 80)
    print()

    print(f"Models to evaluate ({len(MODELS_TO_EVALUATE)}):")
    for i, (model_type, model_variant) in enumerate(MODELS_TO_EVALUATE, 1):
        print(f"  {i}. {model_type} {model_variant}")
    print()

    print(f"Tasks to evaluate ({len(TASKS_TO_EVALUATE)}):")
    for i, task_name in enumerate(TASKS_TO_EVALUATE, 1):
        print(f"  {i}. {task_name}")
    print()

    total_experiments = len(MODELS_TO_EVALUATE) * len(TASKS_TO_EVALUATE)
    print(f"Total experiments to run: {total_experiments}")
    print("=" * 80)
    print()
    print("Starting experiments...")
    print()

    experiment_count = 0
    for model_idx, (model_type, model_variant) in enumerate(MODELS_TO_EVALUATE, 1):
        print()
        print("#" * 80)
        print(f"# Model {model_idx}/{len(MODELS_TO_EVALUATE)}: {model_type} {model_variant}")
        print("#" * 80)
        print()

        for task_idx, task_name in enumerate(TASKS_TO_EVALUATE, 1):
            experiment_count += 1
            print()
            print("=" * 80)
            print(f"Experiment {experiment_count}/{total_experiments}")
            print(f"Model: {model_type} {model_variant}")
            print(f"Task: {task_name}")
            print("=" * 80)
            print()

            try:
                run_token_length_analysis(
                    model_type=model_type,
                    model_variant=model_variant,
                    task_name=task_name,
                    num_examples=5,
                    token_ranges=[(0, 5), (5, 10), (10, 15), (15, 20)],
                    experiment_id="token_length_analysis"
                )
                print()
                print(f"✓ Completed: {model_type} {model_variant} - {task_name}")
                print()
            except Exception as e:
                print()
                print(f"✗ Error in {model_type} {model_variant} - {task_name}:")
                print(f"  {e}")
                import traceback
                traceback.print_exc()
                print()
                print("Continuing with next experiment...")
                print()

    print()
    print("=" * 80)
    print("All experiments completed!")
    print("=" * 80)
    print()
    print("To visualize all results, run:")
    print("  python -m scripts.experiments.visualize_token_length_analysis --experiment-id token_length_analysis")
    print()


if __name__ == "__main__":
    main()
