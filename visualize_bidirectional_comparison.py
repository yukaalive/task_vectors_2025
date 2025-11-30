"""
Visualize bidirectional averaged task vector results with comparison to original TV.
Creates:
1. Bar chart comparing ICL, Original TV, and Averaged TV
2. Text file with top 10 and bottom 10 predictions by COMET score
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results():
    """Load both bidirectional and original results."""
    # Load bidirectional averaged results
    bidirectional_path = Path("outputs/results/main/bidirectional_avg/llama_13B.pkl")
    with open(bidirectional_path, 'rb') as f:
        bidirectional_results = pickle.load(f)

    # Load original results
    original_path = Path("outputs/results/main/camera_ready/llama.pkl")
    with open(original_path, 'rb') as f:
        original_results = pickle.load(f)

    return bidirectional_results, original_results


def create_comparison_graph(bidirectional_results, original_results, save_dir):
    """Create bar chart comparing ICL, Original TV, and Averaged TV with COMET and chrF."""
    exp_key = 'translation_en_ja_easy_and_translation_ja_en_easy'
    bidir = bidirectional_results[exp_key]

    # Get original TV results
    orig_en_ja = original_results.get('translation_en_ja_easy', {})
    orig_ja_en = original_results.get('translation_ja_en_easy', {})

    # Prepare data
    tasks = ['en→ja\n(translation_en_ja_easy)', 'ja→en\n(translation_ja_en_easy)']

    # COMET scores
    icl_comet_scores = [
        bidir.get('task1_icl_comet', 0.0),
        bidir.get('task2_icl_comet', 0.0)
    ]

    original_tv_comet_scores = [
        orig_en_ja.get('tv_comet', 0.0),
        orig_ja_en.get('tv_comet', 0.0)
    ]

    averaged_tv_comet_scores = [
        bidir.get('task1_avg_tv_comet', 0.0),
        bidir.get('task2_avg_tv_comet', 0.0)
    ]

    # chrF scores
    icl_chrf_scores = [
        bidir.get('task1_icl_chrf', 0.0),
        bidir.get('task2_icl_chrf', 0.0)
    ]

    original_tv_chrf_scores = [
        orig_en_ja.get('tv_chrf', 0.0),
        orig_ja_en.get('tv_chrf', 0.0)
    ]

    averaged_tv_chrf_scores = [
        bidir.get('task1_avg_tv_chrf', 0.0),
        bidir.get('task2_avg_tv_chrf', 0.0)
    ]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    x = np.arange(len(tasks))
    width = 0.25

    # Use colors from existing visualize code
    colors = ['steelblue', 'coral', '#2ca02c']

    # --- COMET subplot ---
    bars1 = ax1.bar(x - width, icl_comet_scores, width, label='ICL',
                   color=colors[0], alpha=0.8)
    bars2 = ax1.bar(x, original_tv_comet_scores, width, label='Original TV',
                   color=colors[1], alpha=0.8)
    bars3 = ax1.bar(x + width, averaged_tv_comet_scores, width, label='Bidirectional Averaged TV',
                   color=colors[2], alpha=0.8)

    # Add value labels on bars
    def add_value_labels(ax, bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    add_value_labels(ax1, bars1)
    add_value_labels(ax1, bars2)
    add_value_labels(ax1, bars3)

    # Styling for COMET
    ax1.set_ylabel('COMET Score', fontsize=10)
    ax1.set_title('Comparison: ICL vs Original TV vs Bidirectional Averaged TV (llama_13B)',
                 fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, fontsize=10)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # Remove spines
    for spine in ax1.spines.values():
        spine.set_visible(False)

    # Add retention percentages for COMET
    for i, task in enumerate(['en→ja', 'ja→en']):
        icl = icl_comet_scores[i]
        orig_tv = original_tv_comet_scores[i]
        avg_tv = averaged_tv_comet_scores[i]

        if icl > 0:
            orig_ret = (orig_tv / icl) * 100
            avg_ret = (avg_tv / icl) * 100

            ax1.text(i, -0.12, f'Original: {orig_ret:.1f}%\nAveraged: {avg_ret:.1f}%',
                   ha='center', va='top', fontsize=8,
                   transform=ax1.get_xaxis_transform())

    # --- chrF subplot ---
    bars4 = ax2.bar(x - width, icl_chrf_scores, width, label='ICL',
                   color=colors[0])
    bars5 = ax2.bar(x, original_tv_chrf_scores, width, label='Original TV',
                   color=colors[1])
    bars6 = ax2.bar(x + width, averaged_tv_chrf_scores, width, label='Bidirectional Averaged TV',
                   color=colors[2])

    add_value_labels(ax2, bars4)
    add_value_labels(ax2, bars5)
    add_value_labels(ax2, bars6)

    # Styling for chrF
    ax2.set_ylabel('chrF Score', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tasks, fontsize=10)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # Remove spines
    for spine in ax2.spines.values():
        spine.set_visible(False)

    # Add retention percentages for chrF
    for i, task in enumerate(['en→ja', 'ja→en']):
        icl = icl_chrf_scores[i]
        orig_tv = original_tv_chrf_scores[i]
        avg_tv = averaged_tv_chrf_scores[i]

        if icl > 0:
            orig_ret = (orig_tv / icl) * 100 if orig_tv > 0 else 0.0
            avg_ret = (avg_tv / icl) * 100 if avg_tv > 0 else 0.0

            ax2.text(i, -0.12, f'Original: {orig_ret:.1f}%\nAveraged: {avg_ret:.1f}%',
                   ha='center', va='top', fontsize=8,
                   transform=ax2.get_xaxis_transform())

    plt.tight_layout()

    # Save
    save_path = save_dir / "comparison_icl_original_averaged_tv.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved to: {save_path}")
    plt.close()


def create_detailed_output_file(bidirectional_results, original_results, save_dir):
    """Create text file with top 10 and bottom 10 predictions."""
    exp_key = 'translation_en_ja_easy_and_translation_ja_en_easy'
    bidir = bidirectional_results[exp_key]

    # Get original TV results
    orig_en_ja = original_results.get('translation_en_ja_easy', {})
    orig_ja_en = original_results.get('translation_ja_en_easy', {})

    output_file = save_dir / "bidirectional_detailed_predictions.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("BIDIRECTIONAL AVERAGED TASK VECTOR - DETAILED PREDICTIONS\n")
        f.write("="*100 + "\n\n")

        # Process both tasks
        for task_num, task_name in [(1, 'translation_en_ja_easy'), (2, 'translation_ja_en_easy')]:
            f.write(f"\n{'='*100}\n")
            f.write(f"TASK: {task_name}\n")
            f.write(f"{'='*100}\n\n")

            # Get prediction examples from bidirectional results
            pred_key = f'task{task_num}_prediction_examples'
            if pred_key not in bidir:
                f.write(f"No prediction examples found for task {task_num}\n")
                continue

            pred_ex = bidir[pred_key]
            sources = pred_ex.get('sources', [])
            references = pred_ex.get('references', [])
            icl_preds = pred_ex.get('icl_predictions', [])
            avg_tv_preds = pred_ex.get('avg_tv_predictions', [])
            icl_scores = pred_ex.get('icl_comet_scores', [])
            avg_tv_scores = pred_ex.get('avg_tv_comet_scores', [])

            # Get original TV predictions
            if task_num == 1:
                orig_results_task = orig_en_ja
            else:
                orig_results_task = orig_ja_en

            orig_pred_ex = orig_results_task.get('prediction_examples', {})
            orig_tv_preds = orig_pred_ex.get('tv_predictions', [])
            orig_tv_scores = orig_pred_ex.get('tv_comet_scores', [])
            orig_tv_chrf_scores = orig_pred_ex.get('tv_chrf_scores', [])

            # Get chrF scores from bidirectional results
            icl_chrf_scores = pred_ex.get('icl_chrf_scores', [])
            avg_tv_chrf_scores = pred_ex.get('avg_tv_chrf_scores', [])

            # Create list of examples with all info
            examples = []
            for i in range(len(sources)):
                example = {
                    'index': i,
                    'source': sources[i],
                    'reference': references[i],
                    'icl_pred': icl_preds[i] if i < len(icl_preds) else 'N/A',
                    'icl_score': icl_scores[i] if i < len(icl_scores) else 0.0,
                    'icl_chrf': icl_chrf_scores[i] if i < len(icl_chrf_scores) else 0.0,
                    'orig_tv_pred': orig_tv_preds[i] if i < len(orig_tv_preds) else 'N/A',
                    'orig_tv_score': orig_tv_scores[i] if i < len(orig_tv_scores) else 0.0,
                    'orig_tv_chrf': orig_tv_chrf_scores[i] if i < len(orig_tv_chrf_scores) else 0.0,
                    'avg_tv_pred': avg_tv_preds[i] if i < len(avg_tv_preds) else 'N/A',
                    'avg_tv_score': avg_tv_scores[i] if i < len(avg_tv_scores) else 0.0,
                    'avg_tv_chrf': avg_tv_chrf_scores[i] if i < len(avg_tv_chrf_scores) else 0.0,
                }
                examples.append(example)

            # Sort by averaged TV COMET score
            examples_sorted = sorted(examples, key=lambda x: x['avg_tv_score'], reverse=True)

            # Top 10
            f.write(f"\n{'-'*100}\n")
            f.write(f"TOP 10 PREDICTIONS (Highest Averaged TV COMET Scores)\n")
            f.write(f"{'-'*100}\n\n")

            for rank, ex in enumerate(examples_sorted[:10], 1):
                f.write(f"[Rank {rank}] Example {ex['index'] + 1}\n")
                f.write(f"  Source:     {ex['source']}\n")
                f.write(f"  Reference:  {ex['reference']}\n")
                f.write(f"  \n")
                f.write(f"  ICL Prediction:        {ex['icl_pred']}\n")
                f.write(f"  ICL COMET:             {ex['icl_score']:.4f}\n")
                f.write(f"  ICL chrF:              {ex['icl_chrf']:.4f}\n")
                f.write(f"  \n")
                f.write(f"  Original TV Prediction: {ex['orig_tv_pred']}\n")
                f.write(f"  Original TV COMET:      {ex['orig_tv_score']:.4f}\n")
                f.write(f"  Original TV chrF:       {ex['orig_tv_chrf']:.4f}\n")
                f.write(f"  \n")
                f.write(f"  Averaged TV Prediction: {ex['avg_tv_pred']}\n")
                f.write(f"  Averaged TV COMET:      {ex['avg_tv_score']:.4f}\n")
                f.write(f"  Averaged TV chrF:       {ex['avg_tv_chrf']:.4f}\n")
                f.write(f"\n")

            # Bottom 10
            f.write(f"\n{'-'*100}\n")
            f.write(f"BOTTOM 10 PREDICTIONS (Lowest Averaged TV COMET Scores)\n")
            f.write(f"{'-'*100}\n\n")

            for rank, ex in enumerate(examples_sorted[-10:][::-1], 1):
                f.write(f"[Rank {len(examples_sorted) - rank + 1} from bottom] Example {ex['index'] + 1}\n")
                f.write(f"  Source:     {ex['source']}\n")
                f.write(f"  Reference:  {ex['reference']}\n")
                f.write(f"  \n")
                f.write(f"  ICL Prediction:        {ex['icl_pred']}\n")
                f.write(f"  ICL COMET:             {ex['icl_score']:.4f}\n")
                f.write(f"  ICL chrF:              {ex['icl_chrf']:.4f}\n")
                f.write(f"  \n")
                f.write(f"  Original TV Prediction: {ex['orig_tv_pred']}\n")
                f.write(f"  Original TV COMET:      {ex['orig_tv_score']:.4f}\n")
                f.write(f"  Original TV chrF:       {ex['orig_tv_chrf']:.4f}\n")
                f.write(f"  \n")
                f.write(f"  Averaged TV Prediction: {ex['avg_tv_pred']}\n")
                f.write(f"  Averaged TV COMET:      {ex['avg_tv_score']:.4f}\n")
                f.write(f"  Averaged TV chrF:       {ex['avg_tv_chrf']:.4f}\n")
                f.write(f"\n")

            # Statistics
            f.write(f"\n{'-'*100}\n")
            f.write(f"STATISTICS\n")
            f.write(f"{'-'*100}\n\n")

            avg_icl = np.mean([ex['icl_score'] for ex in examples])
            avg_orig_tv = np.mean([ex['orig_tv_score'] for ex in examples])
            avg_avg_tv = np.mean([ex['avg_tv_score'] for ex in examples])

            avg_icl_chrf = np.mean([ex['icl_chrf'] for ex in examples])
            avg_orig_tv_chrf = np.mean([ex['orig_tv_chrf'] for ex in examples])
            avg_avg_tv_chrf = np.mean([ex['avg_tv_chrf'] for ex in examples])

            f.write(f"  Average ICL COMET:        {avg_icl:.4f}\n")
            f.write(f"  Average Original TV COMET: {avg_orig_tv:.4f}\n")
            f.write(f"  Average Averaged TV COMET: {avg_avg_tv:.4f}\n")
            f.write(f"  \n")
            f.write(f"  Average ICL chrF:         {avg_icl_chrf:.4f}\n")
            f.write(f"  Average Original TV chrF:  {avg_orig_tv_chrf:.4f}\n")
            f.write(f"  Average Averaged TV chrF:  {avg_avg_tv_chrf:.4f}\n")
            f.write(f"  \n")
            f.write(f"  COMET Retention:\n")
            f.write(f"    Original TV:  {(avg_orig_tv/avg_icl*100):.1f}%\n")
            f.write(f"    Averaged TV:  {(avg_avg_tv/avg_icl*100):.1f}%\n")
            f.write(f"  \n")
            f.write(f"  chrF Retention:\n")
            f.write(f"    Original TV:  {(avg_orig_tv_chrf/avg_icl_chrf*100):.1f}%\n")
            f.write(f"    Averaged TV:  {(avg_avg_tv_chrf/avg_icl_chrf*100):.1f}%\n")
            f.write(f"  \n")
            f.write(f"  Number of examples:        {len(examples)}\n")
            f.write(f"\n")

    print(f"Detailed predictions saved to: {output_file}")


def main():
    """Main function."""
    # Load results
    print("Loading results...")
    bidirectional_results, original_results = load_results()

    # Create save directory
    save_dir = Path("outputs/results/main/bidirectional_avg/figures")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create comparison graph
    print("\nCreating comparison graph...")
    create_comparison_graph(bidirectional_results, original_results, save_dir)

    # Create detailed output file
    print("\nCreating detailed predictions file...")
    create_detailed_output_file(bidirectional_results, original_results, save_dir)

    print("\n✅ All outputs created successfully!")
    print(f"   - Graph: {save_dir}/comparison_icl_original_averaged_tv.png")
    print(f"   - Details: {save_dir}/bidirectional_detailed_predictions.txt")


if __name__ == "__main__":
    main()
