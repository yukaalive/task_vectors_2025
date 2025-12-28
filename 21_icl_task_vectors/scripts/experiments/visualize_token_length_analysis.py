"""
トークン長分析の結果を可視化するスクリプト

token_length_analysis.pyで生成された結果を読み込み、グラフとして可視化します。
"""

import os
import sys
import pickle
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

from scripts.utils import MAIN_RESULTS_DIR


def load_results(results_file: str) -> Dict:
    """結果ファイルを読み込む"""
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file, "rb") as f:
        results = pickle.load(f)

    return results


def plot_accuracy_by_token_length(results: Dict, output_file: str = None):
    """
    トークン長ごとの精度をプロット

    Args:
        results: 分析結果の辞書
        output_file: 出力ファイルパス（Noneの場合は表示のみ）
    """
    categories = []
    icl_accuracies = []
    tv_accuracies = []
    counts = []

    # データを抽出
    for category in sorted(results["categories"].keys()):
        cat_results = results["categories"][category]
        categories.append(category.replace("tokens", ""))
        icl_accuracies.append(cat_results["icl_accuracy"])
        tv_accuracies.append(cat_results["tv_accuracy"])
        counts.append(cat_results["num_datasets"])

    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左側: 精度の比較
    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax1.bar(x - width/2, icl_accuracies, width, label='ICL', alpha=0.8)
    bars2 = ax1.bar(x + width/2, tv_accuracies, width, label='Task Vector', alpha=0.8)

    ax1.set_xlabel('Token Length Range', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy by Token Length', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1])

    # 値をバーの上に表示
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)

    # 右側: データセット数の分布
    bars3 = ax2.bar(categories, counts, alpha=0.8, color='gray')
    ax2.set_xlabel('Token Length Range', fontsize=12)
    ax2.set_ylabel('Number of Datasets', fontsize=12)
    ax2.set_title('Dataset Distribution', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # 値をバーの上に表示
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)

    # タイトルにメタデータを追加
    if "metadata" in results:
        meta = results["metadata"]
        task_name = meta.get('task_name', 'Unknown')
        model_info = f"{meta.get('model_type', '')} {meta.get('model_variant', '')}".strip()
        fig.suptitle(f"Task: {task_name} | Model: {model_info}",
                    fontsize=16, fontweight='bold', y=1.00)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()


def plot_accuracy_difference(results: Dict, output_file: str = None):
    """
    ICLとTask Vectorの精度差をプロット

    Args:
        results: 分析結果の辞書
        output_file: 出力ファイルパス（Noneの場合は表示のみ）
    """
    categories = []
    differences = []

    # データを抽出
    for category in sorted(results["categories"].keys()):
        cat_results = results["categories"][category]
        categories.append(category.replace("tokens", ""))
        diff = cat_results["tv_accuracy"] - cat_results["icl_accuracy"]
        differences.append(diff)

    # プロット
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['green' if d > 0 else 'red' for d in differences]
    bars = ax.bar(categories, differences, alpha=0.8, color=colors)

    ax.set_xlabel('Token Length Range', fontsize=12)
    ax.set_ylabel('Accuracy Difference (TV - ICL)', fontsize=12)
    ax.set_title('Task Vector vs ICL Accuracy Difference by Token Length',
                fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)

    # 値をバーの上/下に表示
    for bar, diff in zip(bars, differences):
        height = bar.get_height()
        va = 'bottom' if height > 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{diff:+.3f}',
                ha='center', va=va, fontsize=10)

    # タイトルにメタデータを追加
    if "metadata" in results:
        meta = results["metadata"]
        task_name = meta.get('task_name', 'Unknown')
        model_info = f"{meta.get('model_type', '')} {meta.get('model_variant', '')}".strip()
        fig.suptitle(f"Task: {task_name} | Model: {model_info}",
                    fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()


def create_all_visualizations(results_file: str, output_dir: str = None):
    """
    すべての可視化を作成

    Args:
        results_file: 結果ファイルのパス
        output_dir: 出力ディレクトリ（Noneの場合は結果と同じディレクトリ）
    """
    # 結果を読み込む
    results = load_results(results_file)

    # 出力ディレクトリの設定
    if output_dir is None:
        output_dir = os.path.dirname(results_file)

    os.makedirs(output_dir, exist_ok=True)

    # ファイル名のベース
    base_name = os.path.splitext(os.path.basename(results_file))[0]

    # 精度比較プロット
    print("Creating accuracy comparison plot...")
    plot_accuracy_by_token_length(
        results,
        output_file=os.path.join(output_dir, f"{base_name}_accuracy.png")
    )

    # 精度差プロット
    print("Creating accuracy difference plot...")
    plot_accuracy_difference(
        results,
        output_file=os.path.join(output_dir, f"{base_name}_difference.png")
    )

    print("\nVisualization completed!")


def visualize_multiple_results(experiment_id: str = "token_length_analysis"):
    """
    実験IDに含まれるすべての結果を可視化

    Args:
        experiment_id: 実験ID
    """
    results_dir = os.path.join(MAIN_RESULTS_DIR, experiment_id)

    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    # すべての.pklファイルを検索
    pkl_files = [f for f in os.listdir(results_dir) if f.endswith('.pkl')]

    if not pkl_files:
        print(f"No result files found in {results_dir}")
        return

    print(f"Found {len(pkl_files)} result file(s)")
    print()

    for pkl_file in sorted(pkl_files):
        print(f"Processing: {pkl_file}")
        results_file = os.path.join(results_dir, pkl_file)

        try:
            create_all_visualizations(results_file, output_dir=results_dir)
        except Exception as e:
            print(f"Error processing {pkl_file}: {e}")

        print()


def main():
    """
    メイン関数

    使用例:
        # 特定の結果ファイルを可視化
        python visualize_token_length_analysis.py <results_file.pkl>

        # 実験IDのすべての結果を可視化
        python visualize_token_length_analysis.py --experiment-id token_length_analysis
    """
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python visualize_token_length_analysis.py <results_file.pkl>")
        print("  python visualize_token_length_analysis.py --experiment-id <experiment_id>")
        print("\nExamples:")
        print("  python visualize_token_length_analysis.py outputs/results/main/token_length_analysis/gemma_2b_translation_ja_en_jesc.pkl")
        print("  python visualize_token_length_analysis.py --experiment-id token_length_analysis")
        sys.exit(1)

    if sys.argv[1] == "--experiment-id":
        if len(sys.argv) < 3:
            print("Error: experiment-id not provided")
            sys.exit(1)
        experiment_id = sys.argv[2]
        visualize_multiple_results(experiment_id)
    else:
        results_file = sys.argv[1]
        create_all_visualizations(results_file)


if __name__ == "__main__":
    main()
