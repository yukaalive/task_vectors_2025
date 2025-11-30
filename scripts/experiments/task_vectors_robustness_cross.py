import os

# Set GPU before importing anything else
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from dotenv import load_dotenv

load_dotenv(".env")

import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm

from core.config import FIGURES_DIR
from core.data.task_helpers import get_task_by_name
from core.models.llm_loading import load_model_and_tokenizer
from core.task_vectors import get_task_hiddens, task_vector_accuracy_by_layer
from core.utils.misc import limit_gpus, seed_everything


# Tasks to evaluate - the 4 specific translation tasks
TASKS_FOR_CROSS_COMPARISON = [
    "translation_en_ja_single",
    "translation_ja_en_single",
    "translation_en_ja_easy",
    "translation_ja_en_easy",
]


def create_task_vectors(model, tokenizer):
    task_vectors = {}

    for task_name in tqdm(TASKS_FOR_CROSS_COMPARISON):
        num_examples = 4

        task = get_task_by_name(tokenizer, task_name)

        # Determine generation mode based on task name
        generation_mode = "single" if "_single" in task_name else "multi"
        max_new_tokens = 1 if generation_mode == "single" else 30

        test_datasets = task.create_datasets(num_datasets=50, num_examples=num_examples)
        dev_datasets = task.create_datasets(num_datasets=50, num_examples=num_examples)

        dev_accuracy_by_layer = task_vector_accuracy_by_layer(
            model, tokenizer, task, dev_datasets, layers_to_test=range(10, 20),
            generation_mode=generation_mode, max_new_tokens=max_new_tokens
        )
        best_intermediate_layer = int(max(dev_accuracy_by_layer, key=dev_accuracy_by_layer.get))

        task_hiddens = get_task_hiddens(model, tokenizer, task, test_datasets)

        task_vectors[task_name] = task_hiddens[:, best_intermediate_layer]

    return task_vectors


def calculate_within_task_distances(task_vectors):
    """Calculate within-task cosine distances for each task"""

    within_task_distances = {}

    for task_name in task_vectors.keys():
        # Calculate within-task distances (same task vectors against each other)
        distances = cdist(task_vectors[task_name], task_vectors[task_name], metric="cosine").flatten()
        distances_tensor = torch.tensor(distances)

        within_task_distances[task_name] = {
            "distances": distances_tensor,
            "mean": distances_tensor.mean().item(),
            "std": distances_tensor.std().item()
        }

        print(f"\n{task_name} (within task):")
        print(f"  Mean cosine distance: {within_task_distances[task_name]['mean']:.4f}")
        print(f"  Std cosine distance: {within_task_distances[task_name]['std']:.4f}")

    return within_task_distances


def plot_cross_comparison(within_task_distances):
    """Create histogram plots comparing within-task distances for specified task pairs"""

    # Define the task pairs to compare (blue, orange)
    task_pairs = [
        ("translation_en_ja_single", "translation_ja_en_single"),
        ("translation_ja_en_easy", "translation_en_ja_easy"),
        ("translation_en_ja_single", "translation_en_ja_easy"),
        ("translation_ja_en_single", "translation_ja_en_easy"),
    ]

    # Find the global min and max across all tasks to set a common x-axis range
    all_distances = torch.cat([data['distances'] for data in within_task_distances.values()])
    x_min = all_distances.min().item()
    x_max = all_distances.max().item()
    # Add some padding
    x_range = (x_min - 0.01, x_max + 0.01)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Within-Task Cosine Distance Comparisons", fontsize=16)

    axs = axs.flatten()

    for idx, (task1, task2) in enumerate(task_pairs):
        # Get within-task distances for both tasks
        task1_distances = within_task_distances[task1]['distances']
        task2_distances = within_task_distances[task2]['distances']

        # Create histograms with fixed range
        axs[idx].hist(task1_distances, bins=50, alpha=0.5, color='blue', density=True,
                     label=task1, range=x_range)
        axs[idx].hist(task2_distances, bins=50, alpha=0.5, color='orange', density=True,
                     label=task2, range=x_range)

        # Set title and labels
        title = f"{task1}\nvs\n{task2}"
        axs[idx].set_title(title, fontsize=10)
        axs[idx].set_xlabel("Cosine Distance")
        axs[idx].set_ylabel("Density")
        axs[idx].set_xlim(x_range)  # Set the same x-axis range for all subplots
        axs[idx].legend()

    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(FIGURES_DIR, "task_vectors_within_task_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {save_path}")
    print(f"X-axis range: {x_range[0]:.4f} to {x_range[1]:.4f}")


def main():
    seed_everything(41)
    limit_gpus([1])  # Use GPU 1 (H100 NVL with more memory)

    model_type, model_variant = "youko", "8B"
    model, tokenizer = load_model_and_tokenizer(model_type, model_variant)

    print("Creating task vectors...")
    task_vectors = create_task_vectors(model, tokenizer)

    print("\nCalculating within-task cosine distances...")
    within_task_distances = calculate_within_task_distances(task_vectors)

    print("\nCreating comparison plots...")
    plot_cross_comparison(within_task_distances)


if __name__ == "__main__":
    main()
