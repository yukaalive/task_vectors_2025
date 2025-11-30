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

import sacrebleu
from transformers import PreTrainedModel, PreTrainedTokenizer

from scripts.utils import MAIN_RESULTS_DIR, main_experiment_results_dir

from core.data.task_helpers import get_all_tasks, get_task_by_name
from core.models.llm_loading import load_model_and_tokenizer
from core.models.utils.inference import hidden_to_logits
from core.analysis.utils import logits_top_tokens
from core.analysis.evaluation import calculate_accuracy_on_datasets
from core.task_vectors import run_icl, run_task_vector, run_cross_task_vector, get_task_hiddens
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
        task_name=task_name,
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

            # Calculate chrF scores
            print("\nCalculating chrF scores...")
            icl_chrf_scores = []
            tv_chrf_scores = []

            for ref, icl_pred, tv_pred in zip(references, icl_predictions, tv_predictions):
                # Calculate chrF for ICL
                icl_chrf_result = sacrebleu.sentence_chrf(
                    icl_pred,
                    [ref],
                    word_order=0,  # Character n-grams only
                    char_order=6   # Default: up to 6-grams
                )
                icl_chrf_scores.append(icl_chrf_result.score / 100.0)

                # Calculate chrF for TV
                tv_chrf_result = sacrebleu.sentence_chrf(
                    tv_pred,
                    [ref],
                    word_order=0,
                    char_order=6
                )
                tv_chrf_scores.append(tv_chrf_result.score / 100.0)

            # Store average chrF scores
            comet_results["icl_chrf"] = sum(icl_chrf_scores) / len(icl_chrf_scores) if icl_chrf_scores else 0.0
            comet_results["tv_chrf"] = sum(tv_chrf_scores) / len(tv_chrf_scores) if tv_chrf_scores else 0.0

            # Store sample predictions for visualization (first 10 examples)
            num_samples = min(10, len(sources))
            comet_results["prediction_samples"] = {
                "sources": sources[:num_samples],
                "references": references[:num_samples],
                "icl_predictions": icl_predictions[:num_samples],
                "tv_predictions": tv_predictions[:num_samples],
                "icl_scores": icl_comet_results.get("comet_scores", [])[:num_samples],
                "tv_scores": tv_comet_results.get("comet_scores", [])[:num_samples],
                "icl_chrf_scores": icl_chrf_scores[:num_samples],
                "tv_chrf_scores": tv_chrf_scores[:num_samples],
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
        source_task_name=source_task_name,
        target_task_name=target_task_name,
    )

    accuracies["cross_task_tv"] = calculate_accuracy_on_datasets(target_task, cross_predictions, target_test_datasets)
    print(f"Cross-task TV accuracy: {accuracies['cross_task_tv']:.2f}")

    # Target task vector (using target task's own task vectors)
    print("\n" + "="*50)
    print(f"Target task vector: {target_task_name}")
    print("="*50)

    target_tv_predictions, target_tv_dev_accuracy_by_layer, target_task_hiddens = run_task_vector(
        model,
        tokenizer,
        target_task,
        target_test_datasets,
        target_dev_datasets,
        generation_mode=target_generation_mode,
        max_new_tokens=max_new_tokens_target,
        task_name=target_task_name,
    )

    accuracies["target_tv"] = calculate_accuracy_on_datasets(target_task, target_tv_predictions, target_test_datasets)
    print(f"Target task vector accuracy: {accuracies['target_tv']:.2f}")

    target_tv_best_layer = int(max(target_tv_dev_accuracy_by_layer, key=target_tv_dev_accuracy_by_layer.get))
    print(f"\n{'='*50}")
    print(f"★ TARGET TASK VECTOR BEST LAYER: {target_tv_best_layer}")
    print(f"{'='*50}\n")

    # Store layer information
    layer_info["source_best_layer"] = source_best_layer
    layer_info["target_best_layer"] = target_best_layer
    layer_info["target_tv_best_layer"] = target_tv_best_layer
    layer_info["source_dev_accuracy_by_layer"] = source_dev_accuracy_by_layer
    layer_info["target_dev_accuracy_by_layer"] = target_dev_accuracy_by_layer
    layer_info["target_tv_dev_accuracy_by_layer"] = target_tv_dev_accuracy_by_layer

    print(f"\n{'='*70}")
    print(f"★ BEST LAYER SUMMARY:")
    print(f"  - Source task best layer: {source_best_layer}")
    print(f"  - Target task best layer (with source vector): {target_best_layer}")
    print(f"  - Target task vector best layer (with target vector): {target_tv_best_layer}")
    print(f"{'='*70}\n")

    # COMET evaluation for translation tasks
    icl_comet_scores = None
    cross_task_tv_comet_scores = None
    target_tv_comet_scores = None
    icl_chrf_scores = None
    cross_task_tv_chrf_scores = None
    target_tv_chrf_scores = None

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

            # Calculate chrF scores
            print("\nCalculating chrF scores...")
            icl_chrf_scores = []
            cross_task_tv_chrf_scores = []
            target_tv_chrf_scores = []

            for ref, icl_pred, cross_pred, target_pred in zip(references, icl_predictions, cross_predictions, target_tv_predictions):
                # Calculate chrF for ICL
                icl_chrf_result = sacrebleu.sentence_chrf(
                    icl_pred,
                    [ref],
                    word_order=0,
                    char_order=6
                )
                icl_chrf_scores.append(icl_chrf_result.score / 100.0)

                # Calculate chrF for cross-task TV
                cross_chrf_result = sacrebleu.sentence_chrf(
                    cross_pred,
                    [ref],
                    word_order=0,
                    char_order=6
                )
                cross_task_tv_chrf_scores.append(cross_chrf_result.score / 100.0)

                # Calculate chrF for target TV
                target_chrf_result = sacrebleu.sentence_chrf(
                    target_pred,
                    [ref],
                    word_order=0,
                    char_order=6
                )
                target_tv_chrf_scores.append(target_chrf_result.score / 100.0)

            # Store average chrF scores
            comet_results["icl_chrf"] = sum(icl_chrf_scores) / len(icl_chrf_scores) if icl_chrf_scores else 0.0
            comet_results["cross_task_tv_chrf"] = sum(cross_task_tv_chrf_scores) / len(cross_task_tv_chrf_scores) if cross_task_tv_chrf_scores else 0.0
            comet_results["target_tv_chrf"] = sum(target_tv_chrf_scores) / len(target_tv_chrf_scores) if target_tv_chrf_scores else 0.0

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
        'icl_chrf_scores': icl_chrf_scores if icl_chrf_scores is not None else None,
        'cross_task_tv_chrf_scores': cross_task_tv_chrf_scores if cross_task_tv_chrf_scores is not None else None,
        'target_tv_chrf_scores': target_tv_chrf_scores if target_tv_chrf_scores is not None else None,
    }

    return accuracies, comet_results, layer_info, task_hiddens, prediction_examples


def evaluate_bidirectional_averaged_task_vector(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task1_name: str,
    task2_name: str,
    num_examples: int
) -> Tuple:
    """
    Evaluate bidirectional averaged task vectors (e.g., en_ja_easy + ja_en_easy).

    Creates task vectors from both tasks using the same number of examples,
    averages them, and tests the averaged vector on both task directions.

    Args:
        model: The language model
        tokenizer: The tokenizer
        task1_name: First task name (e.g., "translation_en_ja_easy")
        task2_name: Second task name (e.g., "translation_ja_en_easy")
        num_examples: Number of examples for ICL

    Returns:
        results: Dictionary containing accuracies, COMET scores, and predictions
    """
    seed_everything(41)
    print(f"\n{'='*80}")
    print(f"BIDIRECTIONAL AVERAGED TASK VECTOR EXPERIMENT")
    print(f"Task 1: {task1_name}")
    print(f"Task 2: {task2_name}")
    print(f"{'='*80}\n")

    results = {}

    # Get both tasks
    task1 = get_task_by_name(tokenizer=tokenizer, task_name=task1_name)
    task2 = get_task_by_name(tokenizer=tokenizer, task_name=task2_name)

    # Determine generation mode
    generation_mode1 = "single" if "_single" in task1_name else "multi"
    generation_mode2 = "single" if "_single" in task2_name else "multi"
    max_new_tokens1 = 1 if generation_mode1 == "single" else 30
    max_new_tokens2 = 1 if generation_mode2 == "single" else 30

    num_test_datasets, num_dev_datasets = 50, 50

    # Create datasets for both tasks
    print(f"Creating datasets for {task1_name}...")
    task1_test_datasets = task1.create_datasets(num_datasets=num_test_datasets, num_examples=num_examples)
    task1_dev_datasets = task1.create_datasets(num_datasets=num_dev_datasets, num_examples=num_examples)

    print(f"Creating datasets for {task2_name}...")
    task2_test_datasets = task2.create_datasets(num_datasets=num_test_datasets, num_examples=num_examples)
    task2_dev_datasets = task2.create_datasets(num_datasets=num_dev_datasets, num_examples=num_examples)

    # Display examples used for creating task vectors
    from core.data.datasets.few_shot_format import FewShotFormat
    few_shot_format = FewShotFormat()

    print(f"\n{'='*80}")
    print(f"EXAMPLES USED FOR CREATING TASK VECTORS (from dev datasets)")
    print(f"{'='*80}\n")

    print(f"\n{task1_name} - Dev Examples (first 3):")
    print("-" * 80)
    for i in range(min(3, len(task1_dev_datasets))):
        dataset = task1_dev_datasets[i]
        print(f"\nExample {i+1}:")
        print(f"  Prompt (formatted):")
        formatted_prompt = few_shot_format.format_dataset(dataset, include_train=True)
        print(f"    {formatted_prompt}")
        print(f"  Input: {dataset.test_input}")
        print(f"  Output: {dataset.test_output}")
        if hasattr(dataset, 'train_inputs') and dataset.train_inputs:
            print(f"  ICL examples used: {len(dataset.train_inputs)}")
            for j, (inp, out) in enumerate(zip(dataset.train_inputs[:2], dataset.train_outputs[:2])):
                print(f"    ICL {j+1}: {inp} -> {out}")

    print(f"\n{task2_name} - Dev Examples (first 3):")
    print("-" * 80)
    for i in range(min(3, len(task2_dev_datasets))):
        dataset = task2_dev_datasets[i]
        print(f"\nExample {i+1}:")
        print(f"  Prompt (formatted):")
        formatted_prompt = few_shot_format.format_dataset(dataset, include_train=True)
        print(f"    {formatted_prompt}")
        print(f"  Input: {dataset.test_input}")
        print(f"  Output: {dataset.test_output}")
        if hasattr(dataset, 'train_inputs') and dataset.train_inputs:
            print(f"  ICL examples used: {len(dataset.train_inputs)}")
            for j, (inp, out) in enumerate(zip(dataset.train_inputs[:2], dataset.train_outputs[:2])):
                print(f"    ICL {j+1}: {inp} -> {out}")

    # Get task hiddens for both tasks
    print(f"\n{'='*80}")
    print(f"COMPUTING TASK VECTORS")
    print(f"{'='*80}\n")

    print(f"Computing task vectors for {task1_name}...")
    task1_hiddens = get_task_hiddens(model, tokenizer, task1, task1_dev_datasets, multi_context=False)
    print(f"  Task vector shape: {task1_hiddens.shape}")

    print(f"Computing task vectors for {task2_name}...")
    task2_hiddens = get_task_hiddens(model, tokenizer, task2, task2_dev_datasets, multi_context=False)
    print(f"  Task vector shape: {task2_hiddens.shape}")

    # Average the task vectors
    print(f"\nAveraging task vectors...")
    import numpy as np
    import torch
    averaged_task_hiddens = (task1_hiddens + task2_hiddens) / 2.0
    print(f"Averaged task vector shape: {averaged_task_hiddens.shape}")

    # Find best layer for each task using dev sets with averaged vector
    from core.task_vectors import modulated_generate
    from core.models.utils.llm_layers import get_layers

    num_layers = len(get_layers(model))

    # Find best layer for task1
    print(f"\nFinding best layer for {task1_name} using averaged vector...")
    task1_dev_acc_by_layer = {}

    use_comet_task1 = (
        "ja" in task1_name.lower() and
        task1_name.startswith("translation_") and
        generation_mode1 == "multi" and
        hasattr(task1, 'evaluate_with_comet')
    )

    for layer_num in range(num_layers):
        predictions = modulated_generate(
            model, tokenizer, task1, task1_dev_datasets,
            task_hiddens=averaged_task_hiddens,
            intermediate_layer=layer_num,
            generation_mode=generation_mode1,
            max_new_tokens=max_new_tokens1
        )

        if use_comet_task1:
            sources = [d.test_input for d in task1_dev_datasets]
            references = [d.test_output for d in task1_dev_datasets]
            comet_result = task1.evaluate_with_comet(sources, predictions, references, task1_name)
            task1_dev_acc_by_layer[layer_num] = comet_result['comet']
        else:
            task1_dev_acc_by_layer[layer_num] = calculate_accuracy_on_datasets(task1, predictions, task1_dev_datasets)

        if layer_num % 5 == 0:
            print(f"  Layer {layer_num}: {task1_dev_acc_by_layer[layer_num]:.4f}")

    task1_best_layer = int(max(task1_dev_acc_by_layer, key=task1_dev_acc_by_layer.get))
    print(f"Best layer for {task1_name}: {task1_best_layer} (score: {task1_dev_acc_by_layer[task1_best_layer]:.4f})")

    # Find best layer for task2
    print(f"\nFinding best layer for {task2_name} using averaged vector...")
    task2_dev_acc_by_layer = {}

    use_comet_task2 = (
        "ja" in task2_name.lower() and
        task2_name.startswith("translation_") and
        generation_mode2 == "multi" and
        hasattr(task2, 'evaluate_with_comet')
    )

    for layer_num in range(num_layers):
        predictions = modulated_generate(
            model, tokenizer, task2, task2_dev_datasets,
            task_hiddens=averaged_task_hiddens,
            intermediate_layer=layer_num,
            generation_mode=generation_mode2,
            max_new_tokens=max_new_tokens2
        )

        if use_comet_task2:
            sources = [d.test_input for d in task2_dev_datasets]
            references = [d.test_output for d in task2_dev_datasets]
            comet_result = task2.evaluate_with_comet(sources, predictions, references, task2_name)
            task2_dev_acc_by_layer[layer_num] = comet_result['comet']
        else:
            task2_dev_acc_by_layer[layer_num] = calculate_accuracy_on_datasets(task2, predictions, task2_dev_datasets)

        if layer_num % 5 == 0:
            print(f"  Layer {layer_num}: {task2_dev_acc_by_layer[layer_num]:.4f}")

    task2_best_layer = int(max(task2_dev_acc_by_layer, key=task2_dev_acc_by_layer.get))
    print(f"Best layer for {task2_name}: {task2_best_layer} (score: {task2_dev_acc_by_layer[task2_best_layer]:.4f})")

    # Test on task 1 with averaged vector
    print(f"\n{'='*80}")
    print(f"TESTING AVERAGED VECTOR ON {task1_name}")
    print(f"{'='*80}\n")

    # Display test prompts
    print(f"Test Prompts (first 3 examples):")
    print("-" * 80)
    for i in range(min(3, len(task1_test_datasets))):
        dataset = task1_test_datasets[i]
        print(f"\nTest Example {i+1}:")
        print(f"  Prompt (formatted with ICL):")
        formatted_prompt = few_shot_format.format_dataset(dataset, include_train=True)
        print(f"    {formatted_prompt}")
        print(f"  Expected output: {dataset.test_output}")
        if hasattr(dataset, 'train_inputs') and dataset.train_inputs:
            print(f"  Number of ICL examples in prompt: {len(dataset.train_inputs)}")

    # ICL predictions for task1
    print(f"\nGenerating ICL predictions for {task1_name}...")
    icl_predictions_task1 = run_icl(model, tokenizer, task1, task1_test_datasets, generation_mode=generation_mode1)

    # Averaged TV predictions for task1
    print(f"\nGenerating Averaged TV predictions for {task1_name}...")
    print(f"  Using averaged task vector at layer {task1_best_layer}")
    from core.task_vectors import modulated_generate
    avg_tv_predictions_task1 = modulated_generate(
        model, tokenizer, task1, task1_test_datasets,
        task_hiddens=averaged_task_hiddens,
        intermediate_layer=task1_best_layer,
        generation_mode=generation_mode1,
        max_new_tokens=max_new_tokens1
    )

    # Calculate accuracies for task1
    results['task1_name'] = task1_name
    results['task1_icl_accuracy'] = calculate_accuracy_on_datasets(task1, icl_predictions_task1, task1_test_datasets)
    results['task1_avg_tv_accuracy'] = calculate_accuracy_on_datasets(task1, avg_tv_predictions_task1, task1_test_datasets)
    results['task1_best_layer'] = task1_best_layer

    # COMET evaluation for task1
    if task1_name.startswith("translation_") and hasattr(task1, 'evaluate_with_comet'):
        sources_task1 = [dataset.test_input for dataset in task1_test_datasets]
        references_task1 = [dataset.test_output for dataset in task1_test_datasets]

        icl_comet_task1 = task1.evaluate_with_comet(sources_task1, icl_predictions_task1, references_task1, task1_name)
        avg_tv_comet_task1 = task1.evaluate_with_comet(sources_task1, avg_tv_predictions_task1, references_task1, task1_name)

        results['task1_icl_comet'] = icl_comet_task1['comet']
        results['task1_avg_tv_comet'] = avg_tv_comet_task1['comet']

        # Calculate chrF scores
        icl_chrf_scores_task1 = []
        avg_tv_chrf_scores_task1 = []
        for ref, icl_pred, avg_tv_pred in zip(references_task1, icl_predictions_task1, avg_tv_predictions_task1):
            # ICL chrF
            icl_chrf_result = sacrebleu.sentence_chrf(icl_pred, [ref], word_order=0, char_order=6)
            icl_chrf_scores_task1.append(icl_chrf_result.score / 100.0)
            # Averaged TV chrF
            avg_tv_chrf_result = sacrebleu.sentence_chrf(avg_tv_pred, [ref], word_order=0, char_order=6)
            avg_tv_chrf_scores_task1.append(avg_tv_chrf_result.score / 100.0)

        results['task1_icl_chrf'] = sum(icl_chrf_scores_task1) / len(icl_chrf_scores_task1) if icl_chrf_scores_task1 else 0.0
        results['task1_avg_tv_chrf'] = sum(avg_tv_chrf_scores_task1) / len(avg_tv_chrf_scores_task1) if avg_tv_chrf_scores_task1 else 0.0

        print(f"\n{task1_name} Results:")
        print(f"  ICL COMET: {results['task1_icl_comet']:.4f}")
        print(f"  Averaged TV COMET: {results['task1_avg_tv_comet']:.4f}")
        print(f"  Retention: {(results['task1_avg_tv_comet'] / results['task1_icl_comet'] * 100):.1f}%")
        print(f"  ICL chrF: {results['task1_icl_chrf']:.4f}")
        print(f"  Averaged TV chrF: {results['task1_avg_tv_chrf']:.4f}")

    # Test on task 2 with averaged vector
    print(f"\n{'='*80}")
    print(f"TESTING AVERAGED VECTOR ON {task2_name}")
    print(f"{'='*80}\n")

    # Display test prompts
    print(f"Test Prompts (first 3 examples):")
    print("-" * 80)
    for i in range(min(3, len(task2_test_datasets))):
        dataset = task2_test_datasets[i]
        print(f"\nTest Example {i+1}:")
        print(f"  Prompt (formatted with ICL):")
        formatted_prompt = few_shot_format.format_dataset(dataset, include_train=True)
        print(f"    {formatted_prompt}")
        print(f"  Expected output: {dataset.test_output}")
        if hasattr(dataset, 'train_inputs') and dataset.train_inputs:
            print(f"  Number of ICL examples in prompt: {len(dataset.train_inputs)}")

    # ICL predictions for task2
    print(f"\nGenerating ICL predictions for {task2_name}...")
    icl_predictions_task2 = run_icl(model, tokenizer, task2, task2_test_datasets, generation_mode=generation_mode2)

    # Averaged TV predictions for task2
    print(f"\nGenerating Averaged TV predictions for {task2_name}...")
    print(f"  Using averaged task vector at layer {task2_best_layer}")
    avg_tv_predictions_task2 = modulated_generate(
        model, tokenizer, task2, task2_test_datasets,
        task_hiddens=averaged_task_hiddens,
        intermediate_layer=task2_best_layer,
        generation_mode=generation_mode2,
        max_new_tokens=max_new_tokens2
    )

    # Calculate accuracies for task2
    results['task2_name'] = task2_name
    results['task2_icl_accuracy'] = calculate_accuracy_on_datasets(task2, icl_predictions_task2, task2_test_datasets)
    results['task2_avg_tv_accuracy'] = calculate_accuracy_on_datasets(task2, avg_tv_predictions_task2, task2_test_datasets)
    results['task2_best_layer'] = task2_best_layer

    # COMET evaluation for task2
    if task2_name.startswith("translation_") and hasattr(task2, 'evaluate_with_comet'):
        sources_task2 = [dataset.test_input for dataset in task2_test_datasets]
        references_task2 = [dataset.test_output for dataset in task2_test_datasets]

        icl_comet_task2 = task2.evaluate_with_comet(sources_task2, icl_predictions_task2, references_task2, task2_name)
        avg_tv_comet_task2 = task2.evaluate_with_comet(sources_task2, avg_tv_predictions_task2, references_task2, task2_name)

        results['task2_icl_comet'] = icl_comet_task2['comet']
        results['task2_avg_tv_comet'] = avg_tv_comet_task2['comet']

        # Calculate chrF scores
        icl_chrf_scores_task2 = []
        avg_tv_chrf_scores_task2 = []
        for ref, icl_pred, avg_tv_pred in zip(references_task2, icl_predictions_task2, avg_tv_predictions_task2):
            # ICL chrF
            icl_chrf_result = sacrebleu.sentence_chrf(icl_pred, [ref], word_order=0, char_order=6)
            icl_chrf_scores_task2.append(icl_chrf_result.score / 100.0)
            # Averaged TV chrF
            avg_tv_chrf_result = sacrebleu.sentence_chrf(avg_tv_pred, [ref], word_order=0, char_order=6)
            avg_tv_chrf_scores_task2.append(avg_tv_chrf_result.score / 100.0)

        results['task2_icl_chrf'] = sum(icl_chrf_scores_task2) / len(icl_chrf_scores_task2) if icl_chrf_scores_task2 else 0.0
        results['task2_avg_tv_chrf'] = sum(avg_tv_chrf_scores_task2) / len(avg_tv_chrf_scores_task2) if avg_tv_chrf_scores_task2 else 0.0

        print(f"\n{task2_name} Results:")
        print(f"  ICL COMET: {results['task2_icl_comet']:.4f}")
        print(f"  Averaged TV COMET: {results['task2_avg_tv_comet']:.4f}")
        print(f"  Retention: {(results['task2_avg_tv_comet'] / results['task2_icl_comet'] * 100):.1f}%")
        print(f"  ICL chrF: {results['task2_icl_chrf']:.4f}")
        print(f"  Averaged TV chrF: {results['task2_avg_tv_chrf']:.4f}")

    # Store prediction examples with individual COMET and chrF scores
    results['task1_prediction_examples'] = {
        'sources': sources_task1 if task1_name.startswith("translation_") else None,
        'references': references_task1 if task1_name.startswith("translation_") else None,
        'icl_predictions': icl_predictions_task1,
        'avg_tv_predictions': avg_tv_predictions_task1,
        'icl_comet_scores': icl_comet_task1.get('comet_scores', []) if task1_name.startswith("translation_") else [],
        'avg_tv_comet_scores': avg_tv_comet_task1.get('comet_scores', []) if task1_name.startswith("translation_") else [],
        'icl_chrf_scores': icl_chrf_scores_task1 if task1_name.startswith("translation_") else [],
        'avg_tv_chrf_scores': avg_tv_chrf_scores_task1 if task1_name.startswith("translation_") else [],
    }

    results['task2_prediction_examples'] = {
        'sources': sources_task2 if task2_name.startswith("translation_") else None,
        'references': references_task2 if task2_name.startswith("translation_") else None,
        'icl_predictions': icl_predictions_task2,
        'avg_tv_predictions': avg_tv_predictions_task2,
        'icl_comet_scores': icl_comet_task2.get('comet_scores', []) if task2_name.startswith("translation_") else [],
        'avg_tv_comet_scores': avg_tv_comet_task2.get('comet_scores', []) if task2_name.startswith("translation_") else [],
        'icl_chrf_scores': icl_chrf_scores_task2 if task2_name.startswith("translation_") else [],
        'avg_tv_chrf_scores': avg_tv_chrf_scores_task2 if task2_name.startswith("translation_") else [],
    }

    # Store dev examples used for creating task vectors
    results['task1_dev_examples'] = {
        'inputs': [d.test_input for d in task1_dev_datasets[:5]],
        'outputs': [d.test_output for d in task1_dev_datasets[:5]],
        'prompts': [few_shot_format.format_dataset(d, include_train=True) for d in task1_dev_datasets[:5]],
    }

    results['task2_dev_examples'] = {
        'inputs': [d.test_input for d in task2_dev_datasets[:5]],
        'outputs': [d.test_output for d in task2_dev_datasets[:5]],
        'prompts': [few_shot_format.format_dataset(d, include_train=True) for d in task2_dev_datasets[:5]],
    }

    # Store test examples (prompts)
    results['task1_test_examples'] = {
        'inputs': [d.test_input for d in task1_test_datasets[:5]],
        'outputs': [d.test_output for d in task1_test_datasets[:5]],
        'prompts': [few_shot_format.format_dataset(d, include_train=True) for d in task1_test_datasets[:5]],
    }

    results['task2_test_examples'] = {
        'inputs': [d.test_input for d in task2_test_datasets[:5]],
        'outputs': [d.test_output for d in task2_test_datasets[:5]],
        'prompts': [few_shot_format.format_dataset(d, include_train=True) for d in task2_test_datasets[:5]],
    }

    # Store additional info
    results['num_examples'] = num_examples
    results['task1_dev_accuracy_by_layer'] = task1_dev_acc_by_layer
    results['task2_dev_accuracy_by_layer'] = task2_dev_acc_by_layer
    results['averaged_task_hiddens_shape'] = averaged_task_hiddens.shape

    print(f"\n{'='*80}")
    print(f"BIDIRECTIONAL AVERAGED TASK VECTOR EXPERIMENT COMPLETED")
    print(f"{'='*80}\n")

    return results


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


def run_bidirectional_averaged_experiment(
    model_type: str,
    model_variant: str,
    task1_name: str,
    task2_name: str,
    num_examples: int = 5,
    experiment_id: str = "",
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
) -> None:
    """
    Run bidirectional averaged task vector experiment.

    Args:
        model_type: Model type (e.g., "llama")
        model_variant: Model variant (e.g., "13B")
        task1_name: First task name (e.g., "translation_en_ja_easy")
        task2_name: Second task name (e.g., "translation_ja_en_easy")
        num_examples: Number of ICL examples
        experiment_id: Experiment ID for saving results
        model: Pre-loaded model (optional)
        tokenizer: Pre-loaded tokenizer (optional)
    """
    # Create results file path
    results_file = get_results_file_path(
        model_type,
        model_variant,
        experiment_id=experiment_id + "_bidirectional_avg" if experiment_id else "bidirectional_avg"
    )
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # Load existing results if available
    if os.path.exists(results_file):
        with open(results_file, "rb") as f:
            results = pickle.load(f)
    else:
        results = {}

    # Create experiment key
    experiment_key = f"{task1_name}_and_{task2_name}"

    if experiment_key in results:
        print(f"Skipping bidirectional averaged experiment: {experiment_key}")
        return

    print("Loading model and tokenizer...")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

    import torch
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch device count: {torch.cuda.device_count()}")

    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(model_type, model_variant)
        limit_gpus([1])

    # Run evaluation
    experiment_results = evaluate_bidirectional_averaged_task_vector(
        model, tokenizer, task1_name, task2_name, num_examples
    )

    # Store results
    results[experiment_key] = experiment_results

    # Save results
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
