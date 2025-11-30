import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

def load_results(results_dir):
    """Load all pickle files from the results directory"""
    results = {}

    for pkl_file in Path(results_dir).glob("*.pkl"):
        model_name = pkl_file.stem  # e.g., "swallow_7B"
        with open(pkl_file, 'rb') as f:
            results[model_name] = pickle.load(f)

    return results

def format_task_name(task_name):
    """Format task name for display - add appropriate suffix for multi-token translation tasks"""
    if not task_name.startswith('translation_'):
        return task_name

    # If it ends with _single, keep as is
    if task_name.endswith('_single'):
        return task_name

    # Check if it already has jesc or easy suffix
    if '_jesc' in task_name:
        # Replace _jesc with _jesc_multi (e.g., translation_ja_en_jesc -> translation_ja_en_jesc_multi)
        return task_name.replace('_jesc', '_jesc_multi')
    elif '_easy' in task_name:
        # Replace _easy with _easy_multi (e.g., translation_ja_en_easy -> translation_ja_en_easy_multi)
        return task_name.replace('_easy', '_easy_multi')
    else:
        # Add _multi for other multi-token tasks
        return task_name + '_multi'

def should_use_comet(task_name):
    """Check if task should use COMET score (tasks with 'ja' in name)"""
    return 'ja' in task_name.lower()

def extract_metrics_to_dataframe(results):
    """Extract metrics into a pandas DataFrame"""
    rows = []

    for model_name, tasks_data in results.items():
        for task_name, metrics in tasks_data.items():
            use_comet = should_use_comet(task_name)

            # Select appropriate metric based on task type
            if use_comet:
                icl_metric = metrics.get('icl_comet', np.nan)
                tv_metric = metrics.get('tv_comet', np.nan)
            else:
                icl_metric = metrics.get('icl_accuracy', np.nan)
                tv_metric = metrics.get('tv_accuracy', np.nan)

            row = {
                'model': model_name,
                'task': task_name,
                'task_display': format_task_name(task_name),
                'use_comet': use_comet,
                'icl_metric': icl_metric,
                'tv_metric': tv_metric,
                'baseline_accuracy': metrics.get('baseline_accuracy', np.nan),
                'num_examples': metrics.get('num_examples', np.nan),
                'icl_comet': metrics.get('icl_comet', np.nan),
                'tv_comet': metrics.get('tv_comet', np.nan),
                'icl_chrf': metrics.get('icl_chrf', np.nan),
                'tv_chrf': metrics.get('tv_chrf', np.nan),
                'icl_accuracy': metrics.get('icl_accuracy', np.nan),
                'tv_accuracy': metrics.get('tv_accuracy', np.nan),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df

def plot_unified_comparison(df, output_dir):
    """Plot unified comparison: ICL vs Task Vector (using COMET for ja tasks, accuracy for others)"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Prepare data
    models = df['model'].unique()
    tasks = df['task'].unique()
    task_display_names = [df[df['task'] == task]['task_display'].values[0] for task in tasks]

    # Plot 1: Grouped bar chart by task
    ax1 = axes[0]
    x = np.arange(len(tasks))
    width = 0.15

    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        icl_values = [model_data[model_data['task'] == task]['icl_metric'].values[0]
                      if len(model_data[model_data['task'] == task]) > 0 else 0
                      for task in tasks]
        tv_values = [model_data[model_data['task'] == task]['tv_metric'].values[0]
                     if len(model_data[model_data['task'] == task]) > 0 else 0
                     for task in tasks]

        ax1.bar(x + i * width * 2 - width/2, icl_values, width, label=f'{model} (ICL)', alpha=0.8)
        ax1.bar(x + i * width * 2 + width/2, tv_values, width, label=f'{model} (TV)', alpha=0.8, hatch='//')

    ax1.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score (COMET for JA tasks, Accuracy for others)', fontsize=12, fontweight='bold')
    ax1.set_title('Performance Comparison: ICL vs Task Vector by Task', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * (len(models) - 1))
    ax1.set_xticklabels(task_display_names, rotation=45, ha='right')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1])

    # Plot 2: Grouped bar chart by model
    ax2 = axes[1]
    x = np.arange(len(models))
    width = 0.08

    for i, (task, task_display) in enumerate(zip(tasks, task_display_names)):
        task_data = df[df['task'] == task]
        icl_values = [task_data[task_data['model'] == model]['icl_metric'].values[0]
                      if len(task_data[task_data['model'] == model]) > 0 else 0
                      for model in models]
        tv_values = [task_data[task_data['model'] == model]['tv_metric'].values[0]
                     if len(task_data[task_data['model'] == model]) > 0 else 0
                     for model in models]

        ax2.bar(x + i * width * 2 - width/2, icl_values, width, label=f'{task_display} (ICL)', alpha=0.8)
        ax2.bar(x + i * width * 2 + width/2, tv_values, width, label=f'{task_display} (TV)', alpha=0.8, hatch='//')

    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score (COMET for JA tasks, Accuracy for others)', fontsize=12, fontweight='bold')
    ax2.set_title('Performance Comparison: ICL vs Task Vector by Model', fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width * (len(tasks) - 1))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'accuracy_comparison.png')}")
    plt.close()

def plot_comet_comparison(df, output_dir):
    """Plot COMET score comparison: ICL vs Task Vector"""
    # Filter only translation tasks with COMET scores
    df_comet = df[df['icl_comet'].notna() & df['tv_comet'].notna()].copy()

    if len(df_comet) == 0:
        print("No COMET scores available to plot.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    models = df_comet['model'].unique()
    tasks = df_comet['task'].unique()
    task_display_names = [df_comet[df_comet['task'] == task]['task_display'].values[0] for task in tasks]

    # Plot 1: Grouped bar chart by task
    ax1 = axes[0]
    x = np.arange(len(tasks))
    width = 0.15

    for i, model in enumerate(models):
        model_data = df_comet[df_comet['model'] == model]
        icl_comet = [model_data[model_data['task'] == task]['icl_comet'].values[0]
                     if len(model_data[model_data['task'] == task]) > 0 else 0
                     for task in tasks]
        tv_comet = [model_data[model_data['task'] == task]['tv_comet'].values[0]
                    if len(model_data[model_data['task'] == task]) > 0 else 0
                    for task in tasks]

        ax1.bar(x + i * width * 2 - width/2, icl_comet, width, label=f'{model} (ICL)', alpha=0.8)
        ax1.bar(x + i * width * 2 + width/2, tv_comet, width, label=f'{model} (TV)', alpha=0.8, hatch='//')

    ax1.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax1.set_ylabel('COMET Score', fontsize=12, fontweight='bold')
    ax1.set_title('COMET Score by Task', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * (len(models) - 1))
    ax1.set_xticklabels(task_display_names, rotation=45, ha='right')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1])

    # Plot 2: Grouped bar chart by model
    ax2 = axes[1]
    x = np.arange(len(models))
    width = 0.08

    for i, (task, task_display) in enumerate(zip(tasks, task_display_names)):
        task_data = df_comet[df_comet['task'] == task]
        icl_comet = [task_data[task_data['model'] == model]['icl_comet'].values[0]
                     if len(task_data[task_data['model'] == model]) > 0 else 0
                     for model in models]
        tv_comet = [task_data[task_data['model'] == model]['tv_comet'].values[0]
                    if len(task_data[task_data['model'] == model]) > 0 else 0
                    for model in models]

        ax2.bar(x + i * width * 2 - width/2, icl_comet, width, label=f'{task_display} (ICL)', alpha=0.8)
        ax2.bar(x + i * width * 2 + width/2, tv_comet, width, label=f'{task_display} (TV)', alpha=0.8, hatch='//')

    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('COMET Score', fontsize=12, fontweight='bold')
    ax2.set_title('COMET Score by Model', fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width * (len(tasks) - 1))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comet_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'comet_comparison.png')}")
    plt.close()

def plot_chrf_comparison(df, output_dir):
    """Plot chrF score comparison: ICL vs Task Vector"""
    # Filter only translation tasks with chrF scores
    df_chrf = df[df['icl_chrf'].notna() & df['tv_chrf'].notna()].copy()

    if len(df_chrf) == 0:
        print("No chrF scores available to plot.")
        return
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    models = df_chrf['model'].unique()
    tasks = df_chrf['task'].unique()
    task_display_names = [df_chrf[df_chrf['task'] == task]['task_display'].values[0] for task in tasks]

    # Plot 1: Grouped bar chart by task
    ax1 = axes[0]
    x = np.arange(len(tasks))
    width = 0.15

    for i, model in enumerate(models):
        model_data = df_chrf[df_chrf['model'] == model]
        icl_chrf = [model_data[model_data['task'] == task]['icl_chrf'].values[0]
                    if len(model_data[model_data['task'] == task]) > 0 else 0
                    for task in tasks]
        tv_chrf = [model_data[model_data['task'] == task]['tv_chrf'].values[0]
                   if len(model_data[model_data['task'] == task]) > 0 else 0
                   for task in tasks]

        ax1.bar(x + i * width * 2 - width/2, icl_chrf, width, label=f'{model} (ICL)', alpha=0.8)
        ax1.bar(x + i * width * 2 + width/2, tv_chrf, width, label=f'{model} (TV)', alpha=0.8, hatch='//')

    ax1.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax1.set_ylabel('chrF Score', fontsize=12, fontweight='bold')
    ax1.set_title('chrF Score: ICL vs Task Vector by Task', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * (len(models) - 1))
    ax1.set_xticklabels(task_display_names, rotation=45, ha='right')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1])

    # Plot 2: Grouped bar chart by model
    ax2 = axes[1]
    x = np.arange(len(models))
    width = 0.08

    for i, (task, task_display) in enumerate(zip(tasks, task_display_names)):
        task_data = df_chrf[df_chrf['task'] == task]
        icl_chrf = [task_data[task_data['model'] == model]['icl_chrf'].values[0]
                    if len(task_data[task_data['model'] == model]) > 0 else 0
                    for model in models]
        tv_chrf = [task_data[task_data['model'] == model]['tv_chrf'].values[0]
                   if len(task_data[task_data['model'] == model]) > 0 else 0
                   for model in models]

        ax2.bar(x + i * width * 2 - width/2, icl_chrf, width, label=f'{task_display} (ICL)', alpha=0.8)
        ax2.bar(x + i * width * 2 + width/2, tv_chrf, width, label=f'{task_display} (TV)', alpha=0.8, hatch='//')

    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('chrF Score', fontsize=12, fontweight='bold')
    ax2.set_title('chrF Score: ICL vs Task Vector by Model', fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width * (len(tasks) - 1))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'chrf_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'chrf_comparison.png')}")
    plt.close()

def plot_heatmaps(df, output_dir):
    """Plot heatmaps for accuracy, COMET, and chrF scores"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    # Prepare pivot tables with display names
    icl_acc_pivot = df.pivot(index='task_display', columns='model', values='icl_accuracy')
    tv_acc_pivot = df.pivot(index='task_display', columns='model', values='tv_accuracy')
    icl_comet_pivot = df.pivot(index='task_display', columns='model', values='icl_comet')
    tv_comet_pivot = df.pivot(index='task_display', columns='model', values='tv_comet')
    icl_chrf_pivot = df.pivot(index='task_display', columns='model', values='icl_chrf')
    tv_chrf_pivot = df.pivot(index='task_display', columns='model', values='tv_chrf')

    # Plot heatmaps
    sns.heatmap(icl_acc_pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=axes[0, 0],
                cbar_kws={'label': 'Accuracy'}, vmin=0, vmax=1)
    axes[0, 0].set_title('ICL: Exact Match Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Model', fontsize=10, fontweight='bold')
    axes[0, 0].set_ylabel('Task', fontsize=10, fontweight='bold')

    sns.heatmap(tv_acc_pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=axes[0, 1],
                cbar_kws={'label': 'Accuracy'}, vmin=0, vmax=1)
    axes[0, 1].set_title('Task Vector: Exact Match Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Model', fontsize=10, fontweight='bold')
    axes[0, 1].set_ylabel('Task', fontsize=10, fontweight='bold')

    sns.heatmap(icl_comet_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1, 0],
                cbar_kws={'label': 'COMET Score'}, vmin=0, vmax=1)
    axes[1, 0].set_title('ICL: COMET Score', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Model', fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel('Task', fontsize=10, fontweight='bold')

    sns.heatmap(tv_comet_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1, 1],
                cbar_kws={'label': 'COMET Score'}, vmin=0, vmax=1)
    axes[1, 1].set_title('Task Vector: COMET Score', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Model', fontsize=10, fontweight='bold')
    axes[1, 1].set_ylabel('Task', fontsize=10, fontweight='bold')

    sns.heatmap(icl_chrf_pivot, annot=True, fmt='.3f', cmap='BuGn', ax=axes[2, 0],
                cbar_kws={'label': 'chrF Score'}, vmin=0, vmax=1)
    axes[2, 0].set_title('ICL: chrF Score', fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel('Model', fontsize=10, fontweight='bold')
    axes[2, 0].set_ylabel('Task', fontsize=10, fontweight='bold')

    sns.heatmap(tv_chrf_pivot, annot=True, fmt='.3f', cmap='BuGn', ax=axes[2, 1],
                cbar_kws={'label': 'chrF Score'}, vmin=0, vmax=1)
    axes[2, 1].set_title('Task Vector: chrF Score', fontsize=12, fontweight='bold')
    axes[2, 1].set_xlabel('Model', fontsize=10, fontweight='bold')
    axes[2, 1].set_ylabel('Task', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmaps.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'heatmaps.png')}")
    plt.close()

def plot_dataset_type_comparison(df, output_dir):
    """Plot comparison of jesc, easy, and single dataset types for translation tasks"""
    # Filter only translation tasks with ja
    translation_tasks = df[df['task'].str.contains('translation_.*_ja|translation_ja_', regex=True)].copy()

    if len(translation_tasks) == 0:
        print("No translation tasks found for dataset type comparison.")
        return

    # Extract dataset type (jesc, easy, single) and direction (ja_en or en_ja)
    def extract_dataset_info(task_name):
        parts = task_name.split('_')
        if len(parts) >= 3:
            # Extract direction (e.g., ja_en or en_ja)
            if 'ja' in parts[1] or 'ja' in parts[2]:
                direction = f"{parts[1]}_{parts[2]}"
            else:
                direction = "unknown"

            # Extract dataset type
            if 'jesc' in task_name:
                dataset_type = 'jesc'
            elif 'easy' in task_name:
                dataset_type = 'easy'
            elif 'single' in task_name:
                dataset_type = 'single'
            else:
                dataset_type = 'unknown'

            return direction, dataset_type
        return None, None

    translation_tasks['direction'], translation_tasks['dataset_type'] = zip(
        *translation_tasks['task'].apply(extract_dataset_info)
    )

    # Remove unknown entries
    translation_tasks = translation_tasks[
        (translation_tasks['direction'] != 'unknown') &
        (translation_tasks['dataset_type'] != 'unknown')
    ]

    if len(translation_tasks) == 0:
        print("No valid translation tasks for dataset type comparison.")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    models = translation_tasks['model'].unique()
    directions = translation_tasks['direction'].unique()
    dataset_types = ['single', 'easy', 'jesc']  # Order for plotting

    colors_icl = {'single': '#1f77b4', 'easy': '#ff7f0e', 'jesc': '#2ca02c'}
    colors_tv = {'single': '#aec7e8', 'easy': '#ffbb78', 'jesc': '#98df8a'}

    # Plot 1: ICL - Grouped by direction and dataset type
    ax1 = axes[0, 0]
    x = np.arange(len(directions))
    width = 0.25

    for i, dtype in enumerate(dataset_types):
        icl_values = []
        for direction in directions:
            subset = translation_tasks[
                (translation_tasks['direction'] == direction) &
                (translation_tasks['dataset_type'] == dtype)
            ]
            if len(subset) > 0:
                icl_values.append(subset['icl_metric'].mean())
            else:
                icl_values.append(0)

        ax1.bar(x + i * width - width, icl_values, width,
                label=dtype.upper(), alpha=0.8, color=colors_icl[dtype])

    ax1.set_xlabel('Translation Direction', fontsize=12, fontweight='bold')
    ax1.set_ylabel('ICL Score (COMET)', fontsize=12, fontweight='bold')
    ax1.set_title('ICL: Dataset Type Comparison (JESC vs Easy vs Single)',
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.replace('_', '→') for d in directions], fontsize=11)
    ax1.legend(title='Dataset Type', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1])

    # Plot 2: Task Vector - Grouped by direction and dataset type
    ax2 = axes[0, 1]

    for i, dtype in enumerate(dataset_types):
        tv_values = []
        for direction in directions:
            subset = translation_tasks[
                (translation_tasks['direction'] == direction) &
                (translation_tasks['dataset_type'] == dtype)
            ]
            if len(subset) > 0:
                tv_values.append(subset['tv_metric'].mean())
            else:
                tv_values.append(0)

        ax2.bar(x + i * width - width, tv_values, width,
                label=dtype.upper(), alpha=0.8, color=colors_tv[dtype], hatch='//')

    ax2.set_xlabel('Translation Direction', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Task Vector Score (COMET)', fontsize=12, fontweight='bold')
    ax2.set_title('Task Vector: Dataset Type Comparison (JESC vs Easy vs Single)',
                  fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.replace('_', '→') for d in directions], fontsize=11)
    ax2.legend(title='Dataset Type', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1])

    # Plot 3: ICL vs TV comparison for each dataset type
    ax3 = axes[1, 0]
    x = np.arange(len(dataset_types))
    width = 0.35

    icl_by_dtype = []
    tv_by_dtype = []
    for dtype in dataset_types:
        subset = translation_tasks[translation_tasks['dataset_type'] == dtype]
        icl_by_dtype.append(subset['icl_metric'].mean() if len(subset) > 0 else 0)
        tv_by_dtype.append(subset['tv_metric'].mean() if len(subset) > 0 else 0)

    ax3.bar(x - width/2, icl_by_dtype, width, label='ICL', alpha=0.8, color='steelblue')
    ax3.bar(x + width/2, tv_by_dtype, width, label='Task Vector', alpha=0.8,
            color='coral', hatch='//')

    ax3.set_xlabel('Dataset Type', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Average Score (COMET)', fontsize=12, fontweight='bold')
    ax3.set_title('ICL vs Task Vector: Average Performance by Dataset Type',
                  fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([d.upper() for d in dataset_types], fontsize=11)
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim([0, 1])

    # Add value labels on bars
    for i, (icl_val, tv_val) in enumerate(zip(icl_by_dtype, tv_by_dtype)):
        ax3.text(i - width/2, icl_val + 0.02, f'{icl_val:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax3.text(i + width/2, tv_val + 0.02, f'{tv_val:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 4: Detailed breakdown by model and dataset type
    ax4 = axes[1, 1]
    x = np.arange(len(models))
    width = 0.25

    for i, dtype in enumerate(dataset_types):
        avg_scores = []
        for model in models:
            subset = translation_tasks[
                (translation_tasks['model'] == model) &
                (translation_tasks['dataset_type'] == dtype)
            ]
            if len(subset) > 0:
                # Average of ICL and TV for combined view
                avg_score = (subset['icl_metric'].mean() + subset['tv_metric'].mean()) / 2
                avg_scores.append(avg_score)
            else:
                avg_scores.append(0)

        ax4.bar(x + i * width - width, avg_scores, width,
                label=dtype.upper(), alpha=0.8)

    ax4.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Average Score (ICL + TV)', fontsize=12, fontweight='bold')
    ax4.set_title('Model Performance by Dataset Type (ICL + TV Average)',
                  fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax4.legend(title='Dataset Type', fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_type_comparison.png'),
                dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'dataset_type_comparison.png')}")
    plt.close()

    # Create a detailed table for dataset type comparison
    summary_data = []
    for model in models:
        for direction in directions:
            for dtype in dataset_types:
                subset = translation_tasks[
                    (translation_tasks['model'] == model) &
                    (translation_tasks['direction'] == direction) &
                    (translation_tasks['dataset_type'] == dtype)
                ]
                if len(subset) > 0:
                    summary_data.append({
                        'model': model,
                        'direction': direction,
                        'dataset_type': dtype,
                        'icl_score': subset['icl_metric'].mean(),
                        'tv_score': subset['tv_metric'].mean(),
                        'improvement': subset['tv_metric'].mean() - subset['icl_metric'].mean()
                    })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'dataset_type_summary.csv'), index=False)
    print(f"Saved: {os.path.join(output_dir, 'dataset_type_summary.csv')}")

def export_prediction_examples(results, output_dir):
    """Export prediction examples with sources, references, predictions, and scores"""
    # Collect all translation tasks with prediction samples
    examples_data = []

    for model_name, tasks_data in results.items():
        for task_name, metrics in tasks_data.items():
            if not task_name.startswith('translation_'):
                continue

            # Check if prediction samples exist
            if 'prediction_samples' not in metrics:
                continue

            samples = metrics['prediction_samples']
            num_samples = len(samples.get('sources', []))

            for i in range(num_samples):
                examples_data.append({
                    'model': model_name,
                    'task': task_name,
                    'example_id': i + 1,
                    'source': samples['sources'][i],
                    'reference': samples['references'][i],
                    'icl_prediction': samples['icl_predictions'][i],
                    'tv_prediction': samples['tv_predictions'][i],
                    'icl_score': samples.get('icl_scores', [None] * num_samples)[i],
                    'tv_score': samples.get('tv_scores', [None] * num_samples)[i],
                })

    if len(examples_data) == 0:
        print("No prediction samples found. Make sure to re-run experiments after updating main.py.")
        return

    # Create DataFrame
    examples_df = pd.DataFrame(examples_data)

    # Save to CSV
    csv_path = os.path.join(output_dir, 'prediction_examples.csv')
    examples_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Saved: {csv_path}")

    # Create formatted text output for each task
    for task_name in examples_df['task'].unique():
        task_examples = examples_df[examples_df['task'] == task_name]

        output_file = os.path.join(output_dir, f'examples_{task_name}.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"{'='*100}\n")
            f.write(f"Translation Examples: {task_name}\n")
            f.write(f"{'='*100}\n\n")

            # Group by example_id to show all models for each example
            for example_id in sorted(task_examples['example_id'].unique()):
                example_group = task_examples[task_examples['example_id'] == example_id]

                # Get source and reference (same for all models)
                source = example_group.iloc[0]['source']
                reference = example_group.iloc[0]['reference']

                f.write(f"Example {example_id}\n")
                f.write(f"{'-'*100}\n")
                f.write(f"Source:     {source}\n")
                f.write(f"Reference:  {reference}\n")
                f.write(f"\n")

                # Show predictions from each model
                for _, row in example_group.iterrows():
                    f.write(f"[{row['model']}]\n")
                    f.write(f"  ICL Prediction: {row['icl_prediction']}")
                    if row['icl_score'] is not None:
                        f.write(f" (COMET: {row['icl_score']:.4f})")
                    f.write(f"\n")

                    f.write(f"  TV Prediction:  {row['tv_prediction']}")
                    if row['tv_score'] is not None:
                        f.write(f" (COMET: {row['tv_score']:.4f})")
                    f.write(f"\n\n")

                f.write(f"\n")

        print(f"Saved: {output_file}")

    # Create HTML output with better formatting
    html_path = os.path.join(output_dir, 'prediction_examples.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Translation Prediction Examples</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .task-section { margin-bottom: 40px; border: 2px solid #333; padding: 20px; }
        .task-title { font-size: 24px; font-weight: bold; margin-bottom: 20px; }
        .example { margin-bottom: 30px; padding: 15px; background-color: #f9f9f9; border-left: 4px solid #4CAF50; }
        .example-header { font-weight: bold; font-size: 18px; margin-bottom: 10px; }
        .source { color: #1976D2; font-weight: bold; margin: 5px 0; }
        .reference { color: #388E3C; font-weight: bold; margin: 5px 0; }
        .model-section { margin-left: 20px; margin-top: 10px; }
        .model-name { font-weight: bold; color: #D32F2F; }
        .prediction { margin-left: 40px; margin: 5px 0; }
        .icl { color: #1565C0; }
        .tv { color: #6A1B9A; }
        .score { font-size: 12px; color: #666; }
    </style>
</head>
<body>
    <h1>Translation Prediction Examples</h1>
""")

        for task_name in sorted(examples_df['task'].unique()):
            task_examples = examples_df[examples_df['task'] == task_name]

            f.write(f'<div class="task-section">\n')
            f.write(f'<div class="task-title">{task_name}</div>\n')

            for example_id in sorted(task_examples['example_id'].unique()):
                example_group = task_examples[task_examples['example_id'] == example_id]

                source = example_group.iloc[0]['source']
                reference = example_group.iloc[0]['reference']

                f.write(f'<div class="example">\n')
                f.write(f'<div class="example-header">Example {example_id}</div>\n')
                f.write(f'<div class="source">Source: {source}</div>\n')
                f.write(f'<div class="reference">Reference: {reference}</div>\n')

                for _, row in example_group.iterrows():
                    f.write(f'<div class="model-section">\n')
                    f.write(f'<span class="model-name">[{row["model"]}]</span><br>\n')

                    f.write(f'<div class="prediction icl">• ICL: {row["icl_prediction"]}')
                    if row['icl_score'] is not None:
                        f.write(f' <span class="score">(COMET: {row["icl_score"]:.4f})</span>')
                    f.write(f'</div>\n')

                    f.write(f'<div class="prediction tv">• TV: {row["tv_prediction"]}')
                    if row['tv_score'] is not None:
                        f.write(f' <span class="score">(COMET: {row["tv_score"]:.4f})</span>')
                    f.write(f'</div>\n')

                    f.write(f'</div>\n')

                f.write(f'</div>\n')

            f.write(f'</div>\n')

        f.write("""
</body>
</html>
""")

    print(f"Saved: {html_path}")

def save_summary_table(df, output_dir):
    """Save summary statistics as CSV and LaTeX"""
    # Summary by task (using display names and unified metric)
    summary_by_task = df.groupby('task_display').agg({
        'icl_metric': ['mean', 'std'],
        'tv_metric': ['mean', 'std'],
        'baseline_accuracy': ['mean', 'std']
    }).round(4)

    # Summary by model
    summary_by_model = df.groupby('model').agg({
        'icl_metric': ['mean', 'std'],
        'tv_metric': ['mean', 'std'],
        'baseline_accuracy': ['mean', 'std']
    }).round(4)

    # Save to CSV
    summary_by_task.to_csv(os.path.join(output_dir, 'summary_by_task.csv'))
    summary_by_model.to_csv(os.path.join(output_dir, 'summary_by_model.csv'))
    df.to_csv(os.path.join(output_dir, 'all_results.csv'), index=False)

    print(f"Saved: {os.path.join(output_dir, 'summary_by_task.csv')}")
    print(f"Saved: {os.path.join(output_dir, 'summary_by_model.csv')}")
    print(f"Saved: {os.path.join(output_dir, 'all_results.csv')}")

    # Save to LaTeX
    with open(os.path.join(output_dir, 'summary_by_task.tex'), 'w') as f:
        f.write(summary_by_task.to_latex())
    with open(os.path.join(output_dir, 'summary_by_model.tex'), 'w') as f:
        f.write(summary_by_model.to_latex())

    print(f"Saved: {os.path.join(output_dir, 'summary_by_task.tex')}")
    print(f"Saved: {os.path.join(output_dir, 'summary_by_model.tex')}")

def main():
    # Configuration
    results_dir = "/home/yukaalive/2025workspace/task_vectors/21_icl_task_vectors/outputs/results/main/camera_ready"
    output_dir = "/home/yukaalive/2025workspace/task_vectors/21_icl_task_vectors/outputs/results/main/camera_ready"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print("Loading results...")
    results = load_results(results_dir)
    print(f"Loaded {len(results)} models")

    print("\nExtracting metrics to DataFrame...")
    df = extract_metrics_to_dataframe(results)
    print(f"DataFrame shape: {df.shape}")
    print("\nDataFrame preview:")
    print(df.head(10))

    print("\n" + "="*50)
    print("Generating visualizations...")
    print("="*50)

    # Generate unified plot (only one PNG output)
    plot_unified_comparison(df, output_dir)

    # Generate COMET comparison
    print("\nGenerating COMET comparison...")
    plot_comet_comparison(df, output_dir)

    # Generate chrF comparison
    print("\nGenerating chrF comparison...")
    plot_chrf_comparison(df, output_dir)

    # Generate heatmaps
    print("\nGenerating heatmaps...")
    plot_heatmaps(df, output_dir)

    # Generate dataset type comparison (jesc vs easy vs single)
    print("\nGenerating dataset type comparison...")
    plot_dataset_type_comparison(df, output_dir)

    # Export prediction examples
    print("\nExporting prediction examples...")
    export_prediction_examples(results, output_dir)

    # Save summary tables
    print("\nSaving summary tables...")
    save_summary_table(df, output_dir)

    print("\n" + "="*50)
    print("Visualization complete!")
    print(f"All outputs saved to: {output_dir}")
    print("="*50)

if __name__ == "__main__":
    main()
