"""
Visualize COMET scores for LLaMA and Youko models across all tasks
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

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


def extract_comet_scores(data):
    """Extract COMET scores from data"""
    scores = {}
    for task_name, task_data in data.items():
        scores[task_name] = {
            'icl_comet': task_data.get('icl_comet', 0.0),
            'tv_comet': task_data.get('tv_comet', 0.0),
            'icl_chrf': task_data.get('icl_chrf', 0.0),
            'tv_chrf': task_data.get('tv_chrf', 0.0)
        }
    return scores


def plot_comet_comparison(all_scores, save_dir):
    """Plot COMET score comparison across models and tasks"""
    # Get all unique tasks across all models
    all_tasks = set()
    for model_scores in all_scores.values():
        all_tasks.update(model_scores.keys())
    all_tasks = sorted(all_tasks)

    # Prepare data
    models = list(all_scores.keys())
    n_tasks = len(all_tasks)
    n_models = len(models)

    # Create figure with two subplots (ICL and TV)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    x = np.arange(n_tasks)
    width = 0.25

    colors = {
        'llama': 'steelblue',
        'llama_13B': 'steelblue',
        'llama_7B': 'coral',
        'youko_8B': '#2ca02c'
    }

    # Plot ICL COMET scores
    for i, model in enumerate(models):
        icl_scores = [all_scores[model].get(task, {}).get('icl_comet', 0.0)
                      for task in all_tasks]
        offset = (i - n_models/2 + 0.5) * width
        bars = ax1.bar(x + offset, icl_scores, width,
                       label=model, color=colors.get(model, 'gray'),
                       alpha=0.8, linewidth=1.5)

        # Add value labels on bars
        for bar, val in zip(bars, icl_scores):
            if val > 0:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax1.set_ylabel('COMET Score', fontsize=12, fontweight='bold')
    ax1.set_title('ICL COMET Scores', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([task.replace('translation_', '').replace('_', '\n')
                         for task in all_tasks], rotation=45, ha='right')
    ax1.legend(fontsize=11)
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)

    # Plot TV COMET scores
    for i, model in enumerate(models):
        tv_scores = [all_scores[model].get(task, {}).get('tv_comet', 0.0)
                     for task in all_tasks]
        offset = (i - n_models/2 + 0.5) * width
        bars = ax2.bar(x + offset, tv_scores, width,
                       label=model, color=colors.get(model, 'gray'),
                       alpha=0.8, linewidth=1.5)

        # Add value labels on bars
        for bar, val in zip(bars, tv_scores):
            if val > 0:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax2.set_ylabel('COMET Score', fontsize=12, fontweight='bold')
    ax2.set_title('Task Vector COMET Scores', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([task.replace('translation_', '').replace('_', '\n')
                         for task in all_tasks], rotation=45, ha='right')
    ax2.legend(fontsize=11)
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / 'comet_comparison_all_tasks.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_comet_retention(all_scores, save_dir):
    """Plot COMET retention rate (TV/ICL) for each model and task"""
    all_tasks = set()
    for model_scores in all_scores.values():
        all_tasks.update(model_scores.keys())
    all_tasks = sorted(all_tasks)

    models = list(all_scores.keys())
    n_tasks = len(all_tasks)
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(n_tasks)
    width = 0.25

    colors = {
        'llama': 'steelblue',
        'llama_13B': 'steelblue',
        'llama_7B': 'coral',
        'youko_8B': '#2ca02c'
    }

    for i, model in enumerate(models):
        retention_rates = []
        for task in all_tasks:
            if task in all_scores[model]:
                icl = all_scores[model][task].get('icl_comet', 0.0)
                tv = all_scores[model][task].get('tv_comet', 0.0)
                if icl > 0:
                    retention = (tv / icl) * 100
                else:
                    retention = 0
            else:
                retention = 0
            retention_rates.append(retention)

        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, retention_rates, width,
                      label=model, color=colors.get(model, 'gray'),
                      alpha=0.8, linewidth=1.5)

        # Add value labels on bars
        for bar, val in zip(bars, retention_rates):
            if val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.0f}%',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax.set_ylabel('Retention Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('COMET Score Retention (TV / ICL × 100%)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([task.replace('translation_', '').replace('_', '\n')
                        for task in all_tasks], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(110, max([max(r) for r in [[v for v in ret if v > 0] or [0] for ret in [retention_rates]]])))
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / 'comet_retention_all_tasks.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_chrf_comparison(all_scores, save_dir):
    """Plot chrF score comparison across models and tasks"""
    all_tasks = set()
    for model_scores in all_scores.values():
        all_tasks.update(model_scores.keys())
    all_tasks = sorted(all_tasks)

    models = list(all_scores.keys())
    n_tasks = len(all_tasks)
    n_models = len(models)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    x = np.arange(n_tasks)
    width = 0.25

    colors = {
        'llama': 'steelblue',
        'llama_13B': 'steelblue',
        'llama_7B': 'coral',
        'youko_8B': '#2ca02c'
    }

    # Plot ICL chrF scores
    for i, model in enumerate(models):
        icl_scores = [all_scores[model].get(task, {}).get('icl_chrf', 0.0)
                      for task in all_tasks]
        offset = (i - n_models/2 + 0.5) * width
        bars = ax1.bar(x + offset, icl_scores, width,
                       label=model, color=colors.get(model, 'gray'),
                       alpha=0.8, linewidth=1.5)

        for bar, val in zip(bars, icl_scores):
            if val > 0:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax1.set_ylabel('chrF Score', fontsize=12, fontweight='bold')
    ax1.set_title('ICL chrF Scores', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([task.replace('translation_', '').replace('_', '\n')
                         for task in all_tasks], rotation=45, ha='right')
    ax1.legend(fontsize=11)
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)

    # Plot TV chrF scores
    for i, model in enumerate(models):
        tv_scores = [all_scores[model].get(task, {}).get('tv_chrf', 0.0)
                     for task in all_tasks]
        offset = (i - n_models/2 + 0.5) * width
        bars = ax2.bar(x + offset, tv_scores, width,
                       label=model, color=colors.get(model, 'gray'),
                       alpha=0.8, linewidth=1.5)

        for bar, val in zip(bars, tv_scores):
            if val > 0:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax2.set_ylabel('chrF Score', fontsize=12, fontweight='bold')
    ax2.set_title('Task Vector chrF Scores', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([task.replace('translation_', '').replace('_', '\n')
                         for task in all_tasks], rotation=45, ha='right')
    ax2.legend(fontsize=11)
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / 'chrf_comparison_all_tasks.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_icl_vs_tv_comparison(all_scores, save_dir):
    """Plot average ICL vs Task Vector COMET scores across all models"""
    # Get all tasks
    all_tasks = set()
    for model_scores in all_scores.values():
        all_tasks.update(model_scores.keys())
    all_tasks = sorted(all_tasks)

    # Calculate average scores across models for each task
    avg_icl_scores = []
    avg_tv_scores = []

    for task in all_tasks:
        icl_vals = []
        tv_vals = []
        for model_scores in all_scores.values():
            if task in model_scores:
                icl_val = model_scores[task].get('icl_comet', 0.0)
                tv_val = model_scores[task].get('tv_comet', 0.0)
                if icl_val > 0:
                    icl_vals.append(icl_val)
                if tv_val > 0:
                    tv_vals.append(tv_val)

        avg_icl = np.mean(icl_vals) if icl_vals else 0.0
        avg_tv = np.mean(tv_vals) if tv_vals else 0.0
        avg_icl_scores.append(avg_icl)
        avg_tv_scores.append(avg_tv)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(all_tasks))
    width = 0.35

    colors_icl = 'steelblue'
    colors_tv = 'coral'

    bars1 = ax.bar(x - width/2, avg_icl_scores, width, label='ICL',
                   color=colors_icl, alpha=0.8, linewidth=1.5)
    bars2 = ax.bar(x + width/2, avg_tv_scores, width, label='Task Vector',
                   color=colors_tv, alpha=0.8, linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars1, avg_icl_scores):
        if val > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    for bar, val in zip(bars2, avg_tv_scores):
        if val > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax.set_ylabel('COMET Score', fontsize=12, fontweight='bold')
    ax.set_title('ICL vs Task Vector COMET Scores (Average across 3 models)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([task.replace('translation_', '').replace('_', '\n')
                       for task in all_tasks], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / 'icl_vs_tv_comet_comparison.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def generate_summary_table(all_scores, save_dir):
    """Generate a summary table of all scores"""
    output_path = save_dir / 'summary_table.txt'

    all_tasks = set()
    for model_scores in all_scores.values():
        all_tasks.update(model_scores.keys())
    all_tasks = sorted(all_tasks)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("COMET AND CHRF SCORES SUMMARY - LLAMA AND YOUKO MODELS\n")
        f.write("=" * 120 + "\n\n")

        for task in all_tasks:
            f.write(f"Task: {task}\n")
            f.write("-" * 120 + "\n")
            f.write(f"{'Model':<15} {'ICL COMET':>12} {'TV COMET':>12} {'Retention':>12} {'ICL chrF':>12} {'TV chrF':>12} {'chrF Ret.':>12}\n")
            f.write("-" * 120 + "\n")

            for model in sorted(all_scores.keys()):
                if task in all_scores[model]:
                    icl_comet = all_scores[model][task].get('icl_comet', 0.0)
                    tv_comet = all_scores[model][task].get('tv_comet', 0.0)
                    icl_chrf = all_scores[model][task].get('icl_chrf', 0.0)
                    tv_chrf = all_scores[model][task].get('tv_chrf', 0.0)

                    retention = (tv_comet / icl_comet * 100) if icl_comet > 0 else 0
                    chrf_retention = (tv_chrf / icl_chrf * 100) if icl_chrf > 0 else 0

                    f.write(f"{model:<15} {icl_comet:>12.4f} {tv_comet:>12.4f} {retention:>11.1f}% "
                           f"{icl_chrf:>12.4f} {tv_chrf:>12.4f} {chrf_retention:>11.1f}%\n")
                else:
                    f.write(f"{model:<15} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>12}\n")

            f.write("\n")

        # Overall statistics
        f.write("=" * 120 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("=" * 120 + "\n\n")

        for model in sorted(all_scores.keys()):
            f.write(f"\n{model}:\n")
            f.write("-" * 60 + "\n")

            icl_comets = [scores['icl_comet'] for scores in all_scores[model].values() if scores.get('icl_comet', 0) > 0]
            tv_comets = [scores['tv_comet'] for scores in all_scores[model].values() if scores.get('tv_comet', 0) > 0]
            icl_chrfs = [scores['icl_chrf'] for scores in all_scores[model].values() if scores.get('icl_chrf', 0) > 0]
            tv_chrfs = [scores['tv_chrf'] for scores in all_scores[model].values() if scores.get('tv_chrf', 0) > 0]

            if icl_comets:
                f.write(f"  Average ICL COMET:  {np.mean(icl_comets):.4f} (±{np.std(icl_comets):.4f})\n")
            if tv_comets:
                f.write(f"  Average TV COMET:   {np.mean(tv_comets):.4f} (±{np.std(tv_comets):.4f})\n")
            if icl_comets and tv_comets:
                avg_retention = np.mean([(tv/icl)*100 for tv, icl in zip(tv_comets, icl_comets)])
                f.write(f"  Average Retention:  {avg_retention:.1f}%\n")
            if icl_chrfs:
                f.write(f"  Average ICL chrF:   {np.mean(icl_chrfs):.4f} (±{np.std(icl_chrfs):.4f})\n")
            if tv_chrfs:
                f.write(f"  Average TV chrF:    {np.mean(tv_chrfs):.4f} (±{np.std(tv_chrfs):.4f})\n")
            f.write(f"  Number of tasks:    {len(all_scores[model])}\n")

    print(f"Saved: {output_path}")


def main():
    """Main function"""
    base_dir = Path(__file__).parent

    # Define models and their pkl files
    models = {
        'llama': base_dir / 'llama.pkl',
        'llama_7B': base_dir / 'llama_7B.pkl',
        'youko_8B': base_dir / 'youko_8B.pkl'
    }

    # Load all results
    print("Loading results...")
    all_scores = {}
    for model_name, pkl_path in models.items():
        print(f"  Loading {model_name}...")
        data = load_results(pkl_path)
        all_scores[model_name] = extract_comet_scores(data)
        print(f"    Found {len(all_scores[model_name])} tasks")

    # Create output directory
    save_dir = base_dir / 'figures_comparison'
    save_dir.mkdir(exist_ok=True)

    print("\nGenerating visualizations...")

    # Generate plots
    plot_comet_comparison(all_scores, save_dir)
    plot_comet_retention(all_scores, save_dir)
    plot_icl_vs_tv_comparison(all_scores, save_dir)
    plot_chrf_comparison(all_scores, save_dir)
    generate_summary_table(all_scores, save_dir)

    print(f"\n✅ All visualizations saved to: {save_dir}")
    print("\nGenerated files:")
    for file in sorted(save_dir.glob('*')):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
