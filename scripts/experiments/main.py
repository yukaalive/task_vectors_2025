import sys
import os

# CRITICAL: Set CUDA_VISIBLE_DEVICES before ANY imports that might load torch
# Use both GPUs, then select GPU 1 (H100) in the code
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from dotenv import load_dotenv
load_dotenv(".env")

import pickle
import time
from typing import Optional, Tuple

from transformers import PreTrainedModel, PreTrainedTokenizer

from scripts.utils import MAIN_RESULTS_DIR, main_experiment_results_dir

from core.data.task_helpers import get_all_tasks, get_task_by_name
from core.models.llm_loading import load_model_and_tokenizer
from core.models.utils.inference import hidden_to_logits
from core.analysis.utils import logits_top_tokens
from core.analysis.evaluation import calculate_accuracy_on_datasets
from core.task_vectors import run_icl, run_task_vector, run_cross_task_vector
from core.utils.misc import limit_gpus, seed_everything
from core.experiments_config import MODELS_TO_EVALUATE, TASKS_TO_EVALUATE


def get_results_file_path(model_type: str, model_variant: str, experiment_id: str = "") -> str:
    return os.path.join(main_experiment_results_dir(experiment_id), f"{model_type}_{model_variant}.pkl")


def evaluate_task(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, task_name: str, num_examples: int) -> None:
    seed_everything(41)
    accuracies = {}
    comet_results = {}

    task = get_task_by_name(tokenizer=tokenizer, task_name=task_name)

    # Determine generation mode based on task name
    generation_mode = "single" if "_single" in task_name else "multi"
    print(f"Generation mode: {generation_mode}")

    # Evaluate baseline
    # print("===========↓Baeline↓===========")
    baseline_datasets = task.create_datasets(num_datasets=100, num_examples=0)
    predictions = run_icl(model, tokenizer, task, baseline_datasets, include_train=False, generation_mode=generation_mode)
    accuracies["baseline"] = calculate_accuracy_on_datasets(task, predictions, baseline_datasets)
    # print("===========↑Baeline↑===========")
    print("\n\n")
    # Evaluate ICL and Task Vector
    # TODO: Change back to 400, 100
    # num_test_datasets, num_dev_datasets = 400, 100
    # print("===========↓Regular ICL↓===========")
    num_test_datasets, num_dev_datasets = 50, 50
    test_datasets = task.create_datasets(num_datasets=num_test_datasets, num_examples=num_examples)
    dev_datasets = task.create_datasets(num_datasets=num_dev_datasets, num_examples=num_examples)
    icl_predictions = run_icl(model, tokenizer, task, test_datasets, generation_mode=generation_mode)
    # print("===========↑Regular ICL↑===========")
    print("\n\n")
    # print("===========↓Task Vectors↓===========")
    # Set max_new_tokens based on generation_mode
    max_new_tokens = 1 if generation_mode == "single" else 30
    tv_predictions, tv_dev_accuracy_by_layer, task_hiddens = run_task_vector(
        model,
        tokenizer,
        task,
        test_datasets,
        dev_datasets,
        generation_mode=generation_mode,
        max_new_tokens=max_new_tokens,
    )
    accuracies["tv_dev_by_layer"] = tv_dev_accuracy_by_layer
    accuracies["icl"] = calculate_accuracy_on_datasets(task, icl_predictions, test_datasets)
    accuracies["tv"] = calculate_accuracy_on_datasets(task, tv_predictions, test_datasets)
    print("===========↑Task Vectors↑===========")
    # Add COMET evaluation for translation tasks
    if task_name.startswith("translation_") and hasattr(task, 'evaluate_with_comet'):
        try:
            # Prepare data for COMET evaluation
            # test_input is always the source language (to be translated)
            # test_output is always the target language (reference translation)
            # For ja→en: test_input=Japanese, test_output=English
            # For en→ja: test_input=English, test_output=Japanese
            sources = [dataset.test_input for dataset in test_datasets]
            references = [dataset.test_output for dataset in test_datasets]

            # Extract translation direction for clearer logging
            task_parts = task_name.split("_")
            if len(task_parts) >= 3:
                src_lang = task_parts[1]
                tgt_lang = task_parts[2]
                print(f"\nCOMET Evaluation for {src_lang}→{tgt_lang} translation")

            # COMET evaluation for ICL predictions
            icl_comet_results = task.evaluate_with_comet(sources, icl_predictions, references, task_name)
            comet_results["icl_comet"] = icl_comet_results["comet"]
            # print(f"ICL COMET Score: {icl_comet_results['comet']:.4f}")

            print("\nEvaluating Task Vector with COMET...")
            tv_comet_results = task.evaluate_with_comet(sources, tv_predictions, references, task_name)
            comet_results["tv_comet"] = tv_comet_results["comet"]

            # Store sample predictions for visualization (first 10 examples)
            num_samples = min(10, len(sources))
            comet_results["prediction_samples"] = {
                "sources": sources[:num_samples],
                "references": references[:num_samples],
                "icl_predictions": icl_predictions[:num_samples],
                "tv_predictions": tv_predictions[:num_samples],
                "icl_scores": icl_comet_results.get("comet_scores", [])[:num_samples],
                "tv_scores": tv_comet_results.get("comet_scores", [])[:num_samples],
            }

            # Compare individual COMET scores
            if "comet_scores" in icl_comet_results and "comet_scores" in tv_comet_results:
                # print(f"\nFirst 3 individual COMET scores:")
                for i in range(min(3, len(icl_comet_results["comet_scores"]))):
                    icl_score = icl_comet_results["comet_scores"][i]
                    tv_score = tv_comet_results["comet_scores"][i]
                    # print(f"  Example {i+1}: ICL={icl_score:.4f}, TV={tv_score:.4f}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
    
    tv_ordered_tokens_by_layer = {}
    try:
        for layer_num in tv_dev_accuracy_by_layer.keys():
            task_hidden = task_hiddens.mean(axis=0)[layer_num]
            logits = hidden_to_logits(model, task_hidden)
            tv_ordered_tokens_by_layer[layer_num] = logits_top_tokens(logits, tokenizer, k=100)
    except Exception as e:
        print("Error:", e)

    return accuracies, comet_results, tv_ordered_tokens_by_layer


def evaluate_cross_task(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    source_task_name: str,
    target_task_name: str,
    num_examples: int
) -> Tuple:
    """
    Evaluate cross-task vector transfer.

    Args:
        model: The language model
        tokenizer: The tokenizer
        source_task_name: Source task name (e.g., "translation_ja_en_single")
        target_task_name: Target task name (e.g., "translation_ja_en_easy")
        num_examples: Number of examples for ICL

    Returns:
        accuracies: Dictionary of accuracies
        comet_results: Dictionary of COMET results
        layer_info: Dictionary of layer information
    """
    seed_everything(41)
    accuracies = {}
    comet_results = {}
    layer_info = {}

    # Get tasks
    source_task = get_task_by_name(tokenizer=tokenizer, task_name=source_task_name)
    target_task = get_task_by_name(tokenizer=tokenizer, task_name=target_task_name)

    # Determine generation modes
    source_generation_mode = "single" if "_single" in source_task_name else "multi"
    target_generation_mode = "single" if "_single" in target_task_name else "multi"
    print(f"Source generation mode: {source_generation_mode}")
    print(f"Target generation mode: {target_generation_mode}")

    # Create datasets
    num_test_datasets, num_dev_datasets = 50, 50

    # Source task: use for creating task vectors
    source_dev_datasets = source_task.create_datasets(num_datasets=num_dev_datasets, num_examples=num_examples)

    # Target task: use for evaluation
    target_test_datasets = target_task.create_datasets(num_datasets=num_test_datasets, num_examples=num_examples)
    target_dev_datasets = target_task.create_datasets(num_datasets=num_dev_datasets, num_examples=num_examples)

    # Baseline for target task (no task vector)
    print("Evaluating baseline on target task...")
    baseline_datasets = target_task.create_datasets(num_datasets=100, num_examples=0)
    predictions = run_icl(model, tokenizer, target_task, baseline_datasets, include_train=False, generation_mode=target_generation_mode)
    accuracies["baseline"] = calculate_accuracy_on_datasets(target_task, predictions, baseline_datasets)
    print(f"Baseline accuracy: {accuracies['baseline']:.2f}")

    # Regular ICL on target task (for comparison)
    print("Evaluating regular ICL on target task...")
    icl_predictions = run_icl(model, tokenizer, target_task, target_test_datasets, generation_mode=target_generation_mode)
    accuracies["icl"] = calculate_accuracy_on_datasets(target_task, icl_predictions, target_test_datasets)
    print(f"ICL accuracy: {accuracies['icl']:.2f}")

    # Cross-task vector transfer
    print("\n" + "="*50)
    print(f"Cross-task transfer: {source_task_name} -> {target_task_name}")
    print("="*50)

    max_new_tokens_source = 1 if source_generation_mode == "single" else 30
    max_new_tokens_target = 1 if target_generation_mode == "single" else 30

    cross_predictions, source_best_layer, target_best_layer, source_dev_accuracy_by_layer, target_dev_accuracy_by_layer, source_task_hiddens = run_cross_task_vector(
        model,
        tokenizer,
        source_task,
        target_task,
        source_dev_datasets,
        target_test_datasets,
        target_dev_datasets,
        generation_mode_source=source_generation_mode,
        generation_mode_target=target_generation_mode,
        max_new_tokens_source=max_new_tokens_source,
        max_new_tokens_target=max_new_tokens_target,
    )

    accuracies["cross_task_tv"] = calculate_accuracy_on_datasets(target_task, cross_predictions, target_test_datasets)
    print(f"Cross-task TV accuracy: {accuracies['cross_task_tv']:.2f}")

    # Target task vector (using target task's own task vectors)
    print("\n" + "="*50)
    print(f"Target task vector: {target_task_name}")
    print("="*50)

    # Use layer 13 for multi-token translation (known to be effective)
    target_tv_layers_to_test = [13] if target_generation_mode == "multi" else None

    target_tv_predictions, target_tv_dev_accuracy_by_layer, target_task_hiddens = run_task_vector(
        model,
        tokenizer,
        target_task,
        target_test_datasets,
        target_dev_datasets,
        layers_to_test=target_tv_layers_to_test,
        generation_mode=target_generation_mode,
        max_new_tokens=max_new_tokens_target,
    )

    accuracies["target_tv"] = calculate_accuracy_on_datasets(target_task, target_tv_predictions, target_test_datasets)
    print(f"Target task vector accuracy: {accuracies['target_tv']:.2f}")

    target_tv_best_layer = int(max(target_tv_dev_accuracy_by_layer, key=target_tv_dev_accuracy_by_layer.get))
    print(f"Target task vector best layer: {target_tv_best_layer}")

    # Store layer information
    layer_info["source_best_layer"] = source_best_layer
    layer_info["target_best_layer"] = target_best_layer
    layer_info["target_tv_best_layer"] = target_tv_best_layer
    layer_info["source_dev_accuracy_by_layer"] = source_dev_accuracy_by_layer
    layer_info["target_dev_accuracy_by_layer"] = target_dev_accuracy_by_layer
    layer_info["target_tv_dev_accuracy_by_layer"] = target_tv_dev_accuracy_by_layer

    print(f"\nSource task best layer: {source_best_layer}")
    print(f"Target task best layer (with source vector): {target_best_layer}")
    print(f"Target task vector best layer (with target vector): {target_tv_best_layer}")

    # COMET evaluation for translation tasks
    icl_comet_scores = None
    cross_task_tv_comet_scores = None
    target_tv_comet_scores = None

    if target_task_name.startswith("translation_") and hasattr(target_task, 'evaluate_with_comet'):
        try:
            sources = [dataset.test_input for dataset in target_test_datasets]
            references = [dataset.test_output for dataset in target_test_datasets]

            # COMET for ICL
            icl_comet_results = target_task.evaluate_with_comet(sources, icl_predictions, references, target_task_name)
            comet_results["icl_comet"] = icl_comet_results["comet"]
            icl_comet_scores = icl_comet_results.get("comet_scores", None)

            # COMET for cross-task TV
            print("\nEvaluating Cross-task Task Vector with COMET...")
            cross_comet_results = target_task.evaluate_with_comet(sources, cross_predictions, references, target_task_name)
            comet_results["cross_task_tv_comet"] = cross_comet_results["comet"]
            cross_task_tv_comet_scores = cross_comet_results.get("comet_scores", None)

            # COMET for target TV
            print("\nEvaluating Target Task Vector with COMET...")
            target_tv_comet_results = target_task.evaluate_with_comet(sources, target_tv_predictions, references, target_task_name)
            comet_results["target_tv_comet"] = target_tv_comet_results["comet"]
            target_tv_comet_scores = target_tv_comet_results.get("comet_scores", None)

        except Exception as e:
            import traceback
            traceback.print_exc()

    # Return task hiddens for visualization
    task_hiddens = {
        'source_task_hiddens': source_task_hiddens,
        'target_task_hiddens': target_task_hiddens,
    }

    # Store prediction examples for visualization with COMET scores
    prediction_examples = {
        'sources': [dataset.test_input for dataset in target_test_datasets],
        'references': [dataset.test_output for dataset in target_test_datasets],
        'icl_predictions': icl_predictions,
        'cross_task_tv_predictions': cross_predictions,
        'target_tv_predictions': target_tv_predictions,
        'icl_comet_scores': icl_comet_scores if icl_comet_scores is not None else None,
        'cross_task_tv_comet_scores': cross_task_tv_comet_scores if cross_task_tv_comet_scores is not None else None,
        'target_tv_comet_scores': target_tv_comet_scores if target_tv_comet_scores is not None else None,
    }

    return accuracies, comet_results, layer_info, task_hiddens, prediction_examples


def run_main_experiment(
    model_type: str,
    model_variant: str,
    experiment_id: str = "",
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
) -> None:
    # print("Evaluating model:", model_type, model_variant)
    # 保存先パスを作成
    results_file = get_results_file_path(model_type, model_variant, experiment_id=experiment_id)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    # result_fileが既存にあれば読み込む
    if os.path.exists(results_file):
        with open(results_file, "rb") as f:
            results = pickle.load(f)
    else:
        results = {}

    print("Loading model and tokenizer...")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

    import torch
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch device count: {torch.cuda.device_count()}")

    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(model_type, model_variant)

    print("Loaded model and tokenizer.")
    print(f"Model device: {model.device}")
    print(f"Model dtype: {model.dtype}")

    tasks = get_all_tasks(tokenizer=tokenizer)

    num_examples = 5

    for i, task_name in enumerate(TASKS_TO_EVALUATE):
        task = tasks[task_name]
        if task_name in results:
            print(f"Skipping task {i+1}/{len(tasks)}: {task_name}")
            continue
        results[task_name] = {}

        print("\n" + "=" * 50)
        print(f"Running task {i+1}/{len(tasks)}: {task_name}")

        tic = time.time()
        accuracies, comet_results, tv_ordered_tokens_by_layer = evaluate_task(model, tokenizer, task_name, num_examples)

        # print(f"Baseline Accuracy: {accuracies['baseline']:.2f}")
        # print(f"ICL Accuracy: {accuracies['icl']:.2f}")
        # print(f"Task Vector Accuracy: {accuracies['tv']:.2f}")
        # print(f"Dev Accuracy by layer: ", end="")
        for layer, accuracy in accuracies["tv_dev_by_layer"].items():
            print(f"{layer}: {accuracy:.2f}, ", end="")


        results[task_name] = {
            "baseline_accuracy": accuracies["baseline"],
            "num_examples": num_examples,
            "icl_accuracy": accuracies["icl"],
            "tv_accuracy": accuracies["tv"],
            "tv_dev_accruacy_by_layer": accuracies["tv_dev_by_layer"],
            "tv_ordered_tokens_by_layer": tv_ordered_tokens_by_layer,
            **comet_results,  # Include COMET results
        }

        with open(results_file, "wb") as f:
            pickle.dump(results, f)


def run_cross_task_experiment(
    model_type: str,
    model_variant: str,
    source_task_name: str,
    target_task_name: str,
    experiment_id: str = "",
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
) -> None:
    """
    Run cross-task vector transfer experiment.

    Args:
        model_type: Model type (e.g., "swallow")
        model_variant: Model variant (e.g., "7B")
        source_task_name: Source task name (e.g., "translation_ja_en_single")
        target_task_name: Target task name (e.g., "translation_ja_en_easy")
        experiment_id: Experiment ID for saving results
        model: Pre-loaded model (optional)
        tokenizer: Pre-loaded tokenizer (optional)
    """
    # Create results file path
    results_file = get_results_file_path(
        model_type,
        model_variant,
        experiment_id=experiment_id + "_cross_task" if experiment_id else "cross_task"
    )
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # Load existing results if available
    if os.path.exists(results_file):
        with open(results_file, "rb") as f:
            results = pickle.load(f)
    else:
        results = {}

    # Create cross-task key
    cross_task_key = f"{source_task_name}_to_{target_task_name}"

    if cross_task_key in results:
        print(f"Skipping cross-task experiment: {cross_task_key}")
        return

    print("Loading model and tokenizer...")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

    import torch
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch device count: {torch.cuda.device_count()}")

    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(model_type, model_variant)

    print("Loaded model and tokenizer.")
    print(f"Model device: {model.device}")
    print(f"Model dtype: {model.dtype}")

    num_examples = 5

    print("\n" + "=" * 70)
    print(f"Running cross-task experiment: {source_task_name} -> {target_task_name}")
    print("=" * 70)

    tic = time.time()
    accuracies, comet_results, layer_info, task_hiddens, prediction_examples = evaluate_cross_task(
        model, tokenizer, source_task_name, target_task_name, num_examples
    )

    print(f"\nBaseline Accuracy: {accuracies['baseline']:.2f}")
    print(f"ICL Accuracy: {accuracies['icl']:.2f}")
    print(f"Cross-task TV Accuracy: {accuracies['cross_task_tv']:.2f}")
    print(f"Target TV Accuracy: {accuracies['target_tv']:.2f}")

    print(f"\nSource task dev accuracy by layer: ", end="")
    for layer, accuracy in layer_info["source_dev_accuracy_by_layer"].items():
        print(f"{layer}: {accuracy:.2f}, ", end="")
    print()

    print(f"Target task dev accuracy by layer (with source vector): ", end="")
    for layer, accuracy in layer_info["target_dev_accuracy_by_layer"].items():
        print(f"{layer}: {accuracy:.2f}, ", end="")
    print()

    print(f"Target task dev accuracy by layer (with target vector): ", end="")
    for layer, accuracy in layer_info["target_tv_dev_accuracy_by_layer"].items():
        print(f"{layer}: {accuracy:.2f}, ", end="")
    print()

    toc = time.time()
    print(f"\nTime elapsed: {toc - tic:.2f} seconds")

    results[cross_task_key] = {
        "source_task_name": source_task_name,
        "target_task_name": target_task_name,
        "num_examples": num_examples,
        "baseline_accuracy": accuracies["baseline"],
        "icl_accuracy": accuracies["icl"],
        "cross_task_tv_accuracy": accuracies["cross_task_tv"],
        "target_tv_accuracy": accuracies["target_tv"],
        "source_best_layer": layer_info["source_best_layer"],
        "target_best_layer": layer_info["target_best_layer"],
        "target_tv_best_layer": layer_info["target_tv_best_layer"],
        "source_dev_accuracy_by_layer": layer_info["source_dev_accuracy_by_layer"],
        "target_dev_accuracy_by_layer": layer_info["target_dev_accuracy_by_layer"],
        "target_tv_dev_accuracy_by_layer": layer_info["target_tv_dev_accuracy_by_layer"],
        "source_task_hiddens": task_hiddens["source_task_hiddens"],
        "target_task_hiddens": task_hiddens["target_task_hiddens"],
        "prediction_examples": prediction_examples,
        **comet_results,
    }

    with open(results_file, "wb") as f:
        pickle.dump(results, f)

    print(f"Results saved to: {results_file}")


def get_new_experiment_id() -> str:
    return str(
        max([int(results_dir) for results_dir in os.listdir(MAIN_RESULTS_DIR) if results_dir.isdigit()] + [0]) + 1
    )


def main():
    # Check if cross-task experiment is requested
    if "--cross-task" in sys.argv:
        # Cross-task experiment mode
        # Usage: python main.py --cross-task [model_num or model_type model_variant]

        # Remove --cross-task from argv
        sys.argv.remove("--cross-task")

        # Default cross-task pairs
        CROSS_TASK_PAIRS = [
            ("translation_ja_en_single", "translation_ja_en_easy"),
            # Add more pairs here if needed
        ]

        if len(sys.argv) == 1:
            # Run all models with cross-task experiments
            experiment_id = get_new_experiment_id()
            for model_type, model_variant in MODELS_TO_EVALUATE:
                print("💛💛💛", model_type, model_variant, "💛💛💛")
                model, tokenizer = load_model_and_tokenizer(model_type, model_variant)

                for source_task, target_task in CROSS_TASK_PAIRS:
                    run_cross_task_experiment(
                        model_type, model_variant,
                        source_task, target_task,
                        experiment_id=experiment_id,
                        model=model, tokenizer=tokenizer
                    )
        else:
            # Run specific model with cross-task experiments
            if len(sys.argv) == 2:
                model_num = int(sys.argv[1])
                model_type, model_variant = MODELS_TO_EVALUATE[model_num]
            elif len(sys.argv) == 3:
                model_type, model_variant = sys.argv[1:]

            print("💛💛💛", model_type, model_variant, "💛💛💛")
            model, tokenizer = load_model_and_tokenizer(model_type, model_variant)

            for source_task, target_task in CROSS_TASK_PAIRS:
                run_cross_task_experiment(
                    model_type, model_variant,
                    source_task, target_task,
                    model=model, tokenizer=tokenizer
                )
    else:
        # Original experiment mode (unchanged)
        if len(sys.argv) == 1:
            # Run all models
            # Calculate the experiment_id as the max experiment_id + 1
            experiment_id = get_new_experiment_id()
            for model_type, model_variant in MODELS_TO_EVALUATE:
                print("💛💛💛",model_type,model_variant,"💛💛💛")
                run_main_experiment(model_type, model_variant, experiment_id=experiment_id)
        else:
            if len(sys.argv) == 2:
                model_num = int(sys.argv[1])
                model_type, model_variant = MODELS_TO_EVALUATE[model_num]
            elif len(sys.argv) == 3:
                model_type, model_variant = sys.argv[1:]

            run_main_experiment(model_type, model_variant)


if __name__ == "__main__":
    main()
