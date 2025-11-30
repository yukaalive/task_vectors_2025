"""
Script to view bidirectional averaged task vector experiment results.

Displays:
1. Examples used to create task vectors (from dev datasets)
2. Test prompts used during evaluation
3. Performance metrics
"""
import pickle
import sys
from pathlib import Path


def view_results(pkl_path):
    """View and display results from bidirectional averaged experiment."""
    print(f"Loading results from: {pkl_path}\n")

    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)

    # Get the experiment key
    experiment_keys = list(results.keys())
    if not experiment_keys:
        print("No results found in pickle file.")
        return

    print(f"Found {len(experiment_keys)} experiment(s):")
    for i, key in enumerate(experiment_keys):
        print(f"  {i+1}. {key}")
    print()

    # Display first experiment
    exp_key = experiment_keys[0]
    exp_results = results[exp_key]

    print("="*80)
    print(f"EXPERIMENT: {exp_key}")
    print("="*80)
    print()

    # Display task names
    task1_name = exp_results.get('task1_name', 'Unknown')
    task2_name = exp_results.get('task2_name', 'Unknown')
    print(f"Task 1: {task1_name}")
    print(f"Task 2: {task2_name}")
    print(f"Number of ICL examples: {exp_results.get('num_examples', 'Unknown')}")
    print()

    # Display dev examples (used to create task vectors)
    print("="*80)
    print("DEV EXAMPLES USED TO CREATE TASK VECTORS")
    print("="*80)
    print()

    if 'task1_dev_examples' in exp_results:
        print(f"\n{task1_name} - Dev Examples:")
        print("-"*80)
        dev_ex = exp_results['task1_dev_examples']
        for i in range(min(3, len(dev_ex['inputs']))):
            print(f"\nExample {i+1}:")
            print(f"  Input:  {dev_ex['inputs'][i]}")
            print(f"  Output: {dev_ex['outputs'][i]}")
            print(f"  Full Prompt:")
            print(f"    {dev_ex['prompts'][i]}")

    if 'task2_dev_examples' in exp_results:
        print(f"\n{task2_name} - Dev Examples:")
        print("-"*80)
        dev_ex = exp_results['task2_dev_examples']
        for i in range(min(3, len(dev_ex['inputs']))):
            print(f"\nExample {i+1}:")
            print(f"  Input:  {dev_ex['inputs'][i]}")
            print(f"  Output: {dev_ex['outputs'][i]}")
            print(f"  Full Prompt:")
            print(f"    {dev_ex['prompts'][i]}")

    # Display test examples (prompts used during testing)
    print("\n" + "="*80)
    print("TEST PROMPTS USED DURING EVALUATION")
    print("="*80)
    print()

    if 'task1_test_examples' in exp_results:
        print(f"\n{task1_name} - Test Examples:")
        print("-"*80)
        test_ex = exp_results['task1_test_examples']
        for i in range(min(3, len(test_ex['inputs']))):
            print(f"\nExample {i+1}:")
            print(f"  Input:  {test_ex['inputs'][i]}")
            print(f"  Expected Output: {test_ex['outputs'][i]}")
            print(f"  Full Prompt:")
            print(f"    {test_ex['prompts'][i]}")

    if 'task2_test_examples' in exp_results:
        print(f"\n{task2_name} - Test Examples:")
        print("-"*80)
        test_ex = exp_results['task2_test_examples']
        for i in range(min(3, len(test_ex['inputs']))):
            print(f"\nExample {i+1}:")
            print(f"  Input:  {test_ex['inputs'][i]}")
            print(f"  Expected Output: {test_ex['outputs'][i]}")
            print(f"  Full Prompt:")
            print(f"    {test_ex['prompts'][i]}")

    # Display performance metrics
    print("\n" + "="*80)
    print("PERFORMANCE METRICS")
    print("="*80)
    print()

    # Task 1 results
    print(f"{task1_name}:")
    print(f"  ICL COMET:         {exp_results.get('task1_icl_comet', 0.0):.4f}")
    print(f"  Averaged TV COMET: {exp_results.get('task1_avg_tv_comet', 0.0):.4f}")
    if exp_results.get('task1_icl_comet', 0) > 0:
        retention1 = exp_results.get('task1_avg_tv_comet', 0) / exp_results.get('task1_icl_comet', 1) * 100
        print(f"  Retention:         {retention1:.1f}%")
    print(f"  ICL chrF:          {exp_results.get('task1_icl_chrf', 0.0):.4f}")
    print(f"  Averaged TV chrF:  {exp_results.get('task1_avg_tv_chrf', 0.0):.4f}")
    print(f"  Best Layer:        {exp_results.get('task1_best_layer', 'Unknown')}")
    print()

    # Task 2 results
    print(f"{task2_name}:")
    print(f"  ICL COMET:         {exp_results.get('task2_icl_comet', 0.0):.4f}")
    print(f"  Averaged TV COMET: {exp_results.get('task2_avg_tv_comet', 0.0):.4f}")
    if exp_results.get('task2_icl_comet', 0) > 0:
        retention2 = exp_results.get('task2_avg_tv_comet', 0) / exp_results.get('task2_icl_comet', 1) * 100
        print(f"  Retention:         {retention2:.1f}%")
    print(f"  ICL chrF:          {exp_results.get('task2_icl_chrf', 0.0):.4f}")
    print(f"  Averaged TV chrF:  {exp_results.get('task2_avg_tv_chrf', 0.0):.4f}")
    print(f"  Best Layer:        {exp_results.get('task2_best_layer', 'Unknown')}")
    print()

    # Display sample predictions
    if 'task1_prediction_examples' in exp_results:
        pred_ex = exp_results['task1_prediction_examples']
        if pred_ex['sources']:
            print(f"\n{task1_name} - Sample Predictions:")
            print("-"*80)
            for i in range(min(3, len(pred_ex['sources']))):
                print(f"\nExample {i+1}:")
                print(f"  Source:     {pred_ex['sources'][i]}")
                print(f"  Reference:  {pred_ex['references'][i]}")
                print(f"  ICL:        {pred_ex['icl_predictions'][i]}")
                print(f"  Averaged TV: {pred_ex['avg_tv_predictions'][i]}")

    if 'task2_prediction_examples' in exp_results:
        pred_ex = exp_results['task2_prediction_examples']
        if pred_ex['sources']:
            print(f"\n{task2_name} - Sample Predictions:")
            print("-"*80)
            for i in range(min(3, len(pred_ex['sources']))):
                print(f"\nExample {i+1}:")
                print(f"  Source:     {pred_ex['sources'][i]}")
                print(f"  Reference:  {pred_ex['references'][i]}")
                print(f"  ICL:        {pred_ex['icl_predictions'][i]}")
                print(f"  Averaged TV: {pred_ex['avg_tv_predictions'][i]}")


def main():
    """Main function."""
    if len(sys.argv) > 1:
        pkl_path = sys.argv[1]
    else:
        # Default path
        pkl_path = "outputs/results/main/bidirectional_avg/llama_13B.pkl"

    pkl_path = Path(pkl_path)

    if not pkl_path.exists():
        print(f"Error: File not found: {pkl_path}")
        print("\nUsage: python view_bidirectional_results.py [path_to_pkl_file]")
        return

    view_results(pkl_path)


if __name__ == "__main__":
    main()
