"""
Cross-task experiment results visualization
"""
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
import torch

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_results(pkl_path):
    """Load results from pickle file"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def plot_performance_comparison(results, save_dir):
    """Plot performance comparison: baseline, ICL, cross-task TV, target TV"""
    for task_pair_name, result in results.items():
        # Accuracy comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy - using colors from camera_ready
        methods = ['Baseline', 'ICL', 'Single→Multi Cross TV', 'Multi TV']
        accuracies = [
            float(result['baseline_accuracy']),
            float(result['icl_accuracy']),
            float(result['cross_task_tv_accuracy']),
            float(result.get('target_tv_accuracy', 0.0))
        ]
        colors = ['#d62728', 'steelblue', 'coral', '#2ca02c']

        bars1 = ax1.bar(methods, accuracies, color=colors, alpha=0.8,linewidth=1.5)
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Accuracy Comparison', fontsize=13, fontweight='bold')
        ax1.set_ylim(0, max(accuracies) * 1.2 if max(accuracies) > 0 else 0.5)
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        # COMET scores - using colors from camera_ready
        comet_methods = ['ICL', 'Single→Multi Cross TV', 'Multi TV']
        comet_scores = [
            result['icl_comet'],
            result['cross_task_tv_comet'],
            result.get('target_tv_comet', 0.0)
        ]
        comet_colors = ['steelblue', 'coral', '#2ca02c']

        bars2 = ax2.bar(comet_methods, comet_scores, color=comet_colors, alpha=0.8,linewidth=1.5)
        ax2.set_ylabel('COMET Score', fontsize=12, fontweight='bold')
        ax2.set_title('Translation Quality (COMET)', fontsize=13, fontweight='bold')
        ax2.set_ylim(0, 1.0)
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars2, comet_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        fig.suptitle(f'{result["source_task_name"]} → {result["target_task_name"]}',
                     fontsize=15, fontweight='bold', y=1.02)

        plt.tight_layout()

        # Save
        safe_name = task_pair_name.replace('/', '_')
        save_path = save_dir / f'{safe_name}_performance_comparison.png'
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()


def plot_comet_vs_accuracy(results, save_dir):
    """Scatter plot: COMET score vs Accuracy"""
    fig, ax = plt.subplots(figsize=(8, 8))

    for task_pair_name, result in results.items():
        # ICL - using colors from camera_ready
        icl_acc = float(result['icl_accuracy'])
        icl_comet = result['icl_comet']
        ax.scatter(icl_acc, icl_comet, s=250, alpha=0.8,
                  color='steelblue', marker='o', edgecolors='black', linewidth=2.5,
                  label=f'ICL')

        # Single→Multi Cross TV - using colors from camera_ready
        tv_acc = float(result['cross_task_tv_accuracy'])
        tv_comet = result['cross_task_tv_comet']
        ax.scatter(tv_acc, tv_comet, s=250, alpha=0.8,
                  color='coral', marker='s', edgecolors='black', linewidth=2.5,
                  label=f'Single→Multi Cross TV')

        # Add labels
        ax.annotate('ICL', (icl_acc, icl_comet),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=11, fontweight='bold')
        ax.annotate('Single→Multi Cross TV', (tv_acc, tv_comet),
                   xytext=(10, -15), textcoords='offset points',
                   fontsize=11, fontweight='bold')

    ax.set_xlabel('Accuracy (Exact Match)', fontsize=12, fontweight='bold')
    ax.set_ylabel('COMET Score', fontsize=12, fontweight='bold')
    ax.set_title('Translation Quality vs Accuracy', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.0)

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1.5)

    plt.tight_layout()
    save_path = save_dir / 'comet_vs_accuracy.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def generate_summary_report(results, save_dir):
    """Generate text summary report"""
    report_path = save_dir / 'summary_report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CROSS-TASK VECTOR TRANSFER EXPERIMENT SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        for task_pair_name, result in results.items():
            f.write(f"Task Pair: {task_pair_name}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Source Task: {result['source_task_name']}\n")
            f.write(f"Target Task: {result['target_task_name']}\n")
            f.write(f"Num Examples: {result['num_examples']}\n\n")

            f.write("PERFORMANCE METRICS:\n")
            f.write(f"  Baseline Accuracy:        {float(result['baseline_accuracy']):.4f}\n")
            f.write(f"  ICL Accuracy:             {float(result['icl_accuracy']):.4f}\n")
            f.write(f"  Single→Multi Cross TV Accuracy:   {float(result['cross_task_tv_accuracy']):.4f}\n")
            if 'target_tv_accuracy' in result:
                f.write(f"  Target TV Accuracy:       {float(result['target_tv_accuracy']):.4f}\n")
            f.write("\n")

            f.write("TRANSLATION QUALITY (COMET):\n")
            f.write(f"  ICL COMET:                {result['icl_comet']:.4f}\n")
            f.write(f"  Single→Multi Cross TV COMET:      {result['cross_task_tv_comet']:.4f}\n")
            f.write(f"  COMET Retention (Cross):  {(result['cross_task_tv_comet'] / result['icl_comet'] * 100):.1f}%\n")
            if 'target_tv_comet' in result:
                f.write(f"  Target TV COMET:          {result['target_tv_comet']:.4f}\n")
                f.write(f"  COMET Retention (Target): {(result['target_tv_comet'] / result['icl_comet'] * 100):.1f}%\n")
            f.write("\n")

            f.write("BEST LAYERS:\n")
            f.write(f"  Source Best Layer:        {result['source_best_layer']}\n")
            f.write(f"  Target Best Layer (source vec): {result['target_best_layer']}\n")
            if 'target_tv_best_layer' in result:
                f.write(f"  Target Best Layer (target vec): {result['target_tv_best_layer']}\n")
            f.write("\n")

            # Top 5 source layers
            source_acc = result['source_dev_accuracy_by_layer']
            sorted_source = sorted(source_acc.items(), key=lambda x: x[1], reverse=True)[:5]
            f.write("TOP 5 SOURCE LAYERS:\n")
            for layer, acc in sorted_source:
                f.write(f"  Layer {layer:2d}: {acc:.4f}\n")
            f.write("\n")

            # Target layer accuracies (if any non-zero)
            target_acc = result['target_dev_accuracy_by_layer']
            non_zero_target = [(l, a) for l, a in target_acc.items() if a > 0.0]
            if non_zero_target:
                f.write("NON-ZERO TARGET LAYERS (with source vector):\n")
                for layer, acc in sorted(non_zero_target, key=lambda x: x[1], reverse=True):
                    f.write(f"  Layer {layer:2d}: {acc:.4f}\n")
            else:
                f.write("NOTE: All target layers have 0.00 accuracy (fallback used)\n")

            f.write("\n" + "=" * 80 + "\n\n")

    print(f"Saved: {report_path}")


def plot_task_vectors_tsne(results, save_dir, pkl_path):
    """Plot t-SNE visualization of source and target task vectors"""
    print("\nGenerating t-SNE visualization of task vectors...")

    # Load the full pickle file to get task_hiddens
    with open(pkl_path, 'rb') as f:
        full_data = pickle.load(f)

    for task_pair_name, result in results.items():
        # Check if we have the necessary data
        if 'source_task_hiddens' not in full_data[task_pair_name]:
            print(f"Warning: source_task_hiddens not found in pickle for {task_pair_name}")
            continue
        if 'target_task_hiddens' not in full_data[task_pair_name]:
            print(f"Warning: target_task_hiddens not found in pickle for {task_pair_name}")
            continue

        # Get task hiddens (shape: [num_datasets, num_layers, hidden_size])
        source_task_hiddens = full_data[task_pair_name]['source_task_hiddens']
        target_task_hiddens = full_data[task_pair_name]['target_task_hiddens']

        # Get best layers
        source_best_layer = result['source_best_layer']
        target_tv_best_layer = result.get('target_tv_best_layer', source_best_layer)

        # Extract vectors at best layers
        # source_task_hiddens: (50, num_layers, hidden_size)
        if isinstance(source_task_hiddens, torch.Tensor):
            source_vectors = source_task_hiddens[:, source_best_layer].cpu().numpy()
        else:
            source_vectors = source_task_hiddens[:, source_best_layer]

        if isinstance(target_task_hiddens, torch.Tensor):
            target_vectors = target_task_hiddens[:, target_tv_best_layer].cpu().numpy()
        else:
            target_vectors = target_task_hiddens[:, target_tv_best_layer]

        # Combine vectors
        all_vectors = np.vstack([source_vectors, target_vectors])
        labels = np.array([0] * len(source_vectors) + [1] * len(target_vectors))

        # Apply t-SNE
        print(f"  Applying t-SNE to {all_vectors.shape[0]} vectors of dimension {all_vectors.shape[1]}...")
        tsne = TSNE(n_components=2, random_state=41, perplexity=min(30, len(all_vectors) - 1))
        vectors_2d = tsne.fit_transform(all_vectors)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Source task vectors
        source_2d = vectors_2d[labels == 0]
        ax.scatter(source_2d[:, 0], source_2d[:, 1],
                  c='#1f77b4', s=100, alpha=0.6,
                  label=f'Source Task (layer {source_best_layer})',
                  edgecolors='black', linewidth=0.5)

        # Target task vectors
        target_2d = vectors_2d[labels == 1]
        ax.scatter(target_2d[:, 0], target_2d[:, 1],
                  c='#2ca02c', s=100, alpha=0.6,
                  label=f'Target Task (layer {target_tv_best_layer})',
                  marker='s', edgecolors='black', linewidth=0.5)

        # Add centroids
        source_centroid = np.mean(source_2d, axis=0)
        target_centroid = np.mean(target_2d, axis=0)

        ax.scatter(*source_centroid, c='#1f77b4', s=300, alpha=1.0,
                  marker='*', edgecolors='black', linewidth=2,
                  label='Source Centroid', zorder=10)
        ax.scatter(*target_centroid, c='#2ca02c', s=300, alpha=1.0,
                  marker='*', edgecolors='black', linewidth=2,
                  label='Target Centroid', zorder=10)

        # Add task names as text
        ax.text(source_centroid[0], source_centroid[1] - 2,
               result['source_task_name'],
               fontsize=10, ha='center', va='top', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        ax.text(target_centroid[0], target_centroid[1] - 2,
               result['target_task_name'],
               fontsize=10, ha='center', va='top', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

        ax.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
        ax.set_title(f't-SNE Visualization of Task Vectors\n{result["source_task_name"]} → {result["target_task_name"]}',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        safe_name = task_pair_name.replace('/', '_')
        save_path = save_dir / f'{safe_name}_task_vectors_tsne.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"  Saved: {save_path}")
        plt.close()


def generate_prediction_examples_html(results, save_dir, pkl_path):
    """Generate HTML file with prediction examples comparison"""
    print("\nGenerating prediction examples HTML...")

    # Load the full pickle file to get prediction examples
    with open(pkl_path, 'rb') as f:
        full_data = pickle.load(f)

    for task_pair_name, result in results.items():
        # Check if we have prediction examples
        if 'prediction_examples' not in full_data[task_pair_name]:
            print(f"Warning: prediction_examples not found in pickle for {task_pair_name}")
            continue

        examples = full_data[task_pair_name]['prediction_examples']

        # Get COMET scores
        icl_scores = examples.get('icl_comet_scores', [])
        cross_scores = examples.get('cross_task_tv_comet_scores', [])
        target_scores = examples.get('target_tv_comet_scores', [])

        # Calculate average score for each example
        avg_scores = []
        for i in range(len(examples['sources'])):
            scores = []
            if icl_scores and i < len(icl_scores):
                scores.append(icl_scores[i])
            if cross_scores and i < len(cross_scores):
                scores.append(cross_scores[i])
            if target_scores and i < len(target_scores):
                scores.append(target_scores[i])
            avg_scores.append(np.mean(scores) if scores else 0)

        # Sort by average score
        sorted_indices = np.argsort(avg_scores)

        # Select top 5 (success) and bottom 5 (failure) examples
        num_total = len(sorted_indices)
        if num_total >= 10:
            selected_indices = list(sorted_indices[-5:])[::-1] + list(sorted_indices[:5])
            section_labels = ['🎉 Success Examples (High COMET Scores)'] * 5 + ['⚠️ Failure Examples (Low COMET Scores)'] * 5
        else:
            selected_indices = list(sorted_indices[::-1])
            section_labels = ['Examples'] * len(selected_indices)

        # Create HTML file
        html_path = save_dir / f'{task_pair_name.replace("/", "_")}_prediction_examples.html'

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Translation Prediction Examples</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        h1 { color: #333; text-align: center; }
        h3 { color: #2c3e50; margin-top: 30px; margin-bottom: 15px; text-align: center; }
        .example {
            margin-bottom: 30px;
            padding: 20px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .success-example {
            border-left: 5px solid #27ae60;
        }
        .failure-example {
            border-left: 5px solid #e74c3c;
        }
        .example-header {
            font-weight: bold;
            font-size: 18px;
            margin-bottom: 15px;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .source {
            color: #e74c3c;
            font-weight: bold;
            margin: 10px 0;
            font-size: 16px;
        }
        .reference {
            color: #27ae60;
            font-weight: bold;
            margin: 10px 0;
            font-size: 16px;
        }
        .prediction-section {
            margin-left: 20px;
            margin-top: 15px;
            border-left: 3px solid #95a5a6;
            padding-left: 15px;
        }
        .method-name {
            font-weight: bold;
            color: #34495e;
            font-size: 14px;
            margin-bottom: 5px;
        }
        .score {
            font-size: 12px;
            color: #7f8c8d;
            margin-left: 10px;
        }
        .score-good { color: #27ae60; font-weight: bold; }
        .score-bad { color: #e74c3c; font-weight: bold; }
        .score-medium { color: #f39c12; font-weight: bold; }
        .icl { color: #2980b9; }
        .cross-task { color: #d35400; }
        .target-tv { color: #16a085; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #34495e;
            color: white;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
    </style>
</head>
<body>
    <h1>Translation Prediction Examples</h1>
    <h2 style="text-align: center; color: #7f8c8d;">""")
            f.write(f"{result['source_task_name']} → {result['target_task_name']}")
            f.write("""</h2>
""")

            # Generate examples
            current_section = None
            for idx, i in enumerate(selected_indices):
                # Add section header
                if section_labels[idx] != current_section:
                    current_section = section_labels[idx]
                    f.write(f"<h3>{current_section}</h3>\n")

                # Determine example class
                example_class = "success-example" if "Success" in current_section else "failure-example"

                # Get scores
                icl_score = icl_scores[i] if icl_scores and i < len(icl_scores) else None
                cross_score = cross_scores[i] if cross_scores and i < len(cross_scores) else None
                target_score = target_scores[i] if target_scores and i < len(target_scores) else None

                def score_class(score):
                    if score is None:
                        return ""
                    if score >= 0.7:
                        return "score-good"
                    elif score >= 0.4:
                        return "score-medium"
                    else:
                        return "score-bad"

                def format_score(score):
                    return f"{score:.3f}" if score is not None else "N/A"

                f.write(f"""
    <div class="example {example_class}">
        <div class="example-header">Example {idx+1} (Index: {i})</div>
        <div class="source">Source (Japanese): {examples['sources'][i]}</div>
        <div class="reference">Reference (Ground Truth): {examples['references'][i]}</div>

        <div class="prediction-section">
            <table>
                <tr>
                    <th>Method</th>
                    <th>Prediction</th>
                    <th>COMET Score</th>
                </tr>
                <tr>
                    <td><span class="method-name icl">ICL</span></td>
                    <td class="icl">{examples['icl_predictions'][i]}</td>
                    <td><span class="score {score_class(icl_score)}">{format_score(icl_score)}</span></td>
                </tr>
                <tr>
                    <td><span class="method-name cross-task">Single→Multi Cross TV</span></td>
                    <td class="cross-task">{examples['cross_task_tv_predictions'][i]}</td>
                    <td><span class="score {score_class(cross_score)}">{format_score(cross_score)}</span></td>
                </tr>
                <tr>
                    <td><span class="method-name target-tv">Target TV</span></td>
                    <td class="target-tv">{examples['target_tv_predictions'][i]}</td>
                    <td><span class="score {score_class(target_score)}">{format_score(target_score)}</span></td>
                </tr>
            </table>
        </div>
    </div>
""")

            f.write("""
</body>
</html>
""")

        print(f"  Saved: {html_path}")


def main():
    """Main visualization function"""
    # Paths
    pkl_path = Path(__file__).parent / 'swallow_7B.pkl'
    save_dir = Path(__file__).parent / 'figures'
    save_dir.mkdir(exist_ok=True)

    print(f"Loading results from: {pkl_path}")
    results = load_results(pkl_path)
    print(f"Found {len(results)} task pair(s)\n")

    # Generate all visualizations
    print("Generating visualizations...\n")

    plot_performance_comparison(results, save_dir)
    plot_comet_vs_accuracy(results, save_dir)
    plot_task_vectors_tsne(results, save_dir, pkl_path)
    generate_prediction_examples_html(results, save_dir, pkl_path)
    generate_summary_report(results, save_dir)

    print(f"\n✅ All visualizations saved to: {save_dir}")
    print("\nGenerated files:")
    for file in sorted(save_dir.glob('*')):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
