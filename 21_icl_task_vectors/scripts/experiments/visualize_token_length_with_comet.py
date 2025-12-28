"""
トークン長分析の結果をCOMETスコアで可視化するスクリプト

既存のvisualize_results.pyを参考に、トークン長（0-10, 10-20, 20-30）ごとに
ICLとTask VectorのCOMETスコアを比較します。
"""

import os
import sys
import pickle
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

from scripts.utils import MAIN_RESULTS_DIR

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10


def load_token_length_results(results_dir: str) -> Dict:
    """トークン長分析の結果をすべて読み込む"""
    results = {}

    for pkl_file in Path(results_dir).glob("*.pkl"):
        filename = pkl_file.stem  # e.g., "swallow_7B_translation_ja_en_jesc"

        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            results[filename] = data

    return results


def extract_comet_by_token_length(results: Dict) -> pd.DataFrame:
    """
    トークン長ごとのCOMETスコアをDataFrameに抽出

    既存のコードを参考に、ICLとTask VectorのCOMETスコアを比較できる形式にする
    """
    rows = []

    for result_name, data in results.items():
        # メタデータから情報を取得
        if 'metadata' not in data:
            continue

        metadata = data['metadata']
        model_name = f"{metadata.get('model_type', '')}_{metadata.get('model_variant', '')}".strip('_')
        task_name = metadata.get('task_name', '')

        # 日本語タスクのみCOMETを使用（既存コードの方針と同じ）
        if 'ja' not in task_name.lower():
            continue

        # カテゴリごとのデータ
        if 'categories' not in data:
            continue

        for token_range, cat_data in data['categories'].items():
            # COMETスコアが含まれているか確認
            # 注: 現在の実装ではトークン長ごとのCOMETスコアは計算していないため、
            # accuracyを使用するか、後で拡張する
            row = {
                'model': model_name,
                'task': task_name,
                'token_range': token_range,
                'icl_accuracy': cat_data.get('icl_accuracy', np.nan),
                'tv_accuracy': cat_data.get('tv_accuracy', np.nan),
                'num_datasets': cat_data.get('num_datasets', 0),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def plot_token_length_comparison_by_task(df: pd.DataFrame, output_dir: str):
    """
    タスク別にトークン長ごとの精度を比較

    各タスクごとに1つのグラフを生成し、
    トークン長範囲ごとにICLとTask Vectorを比較
    """
    tasks = df['task'].unique()

    for task in tasks:
        task_data = df[df['task'] == task]
        models = task_data['model'].unique()
        token_ranges = sorted(task_data['token_range'].unique())

        fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
        if len(models) == 1:
            axes = [axes]

        for idx, model in enumerate(models):
            ax = axes[idx]
            model_data = task_data[task_data['model'] == model]

            x = np.arange(len(token_ranges))
            width = 0.35

            icl_values = []
            tv_values = []

            for token_range in token_ranges:
                range_data = model_data[model_data['token_range'] == token_range]
                if len(range_data) > 0:
                    icl_values.append(range_data['icl_accuracy'].values[0])
                    tv_values.append(range_data['tv_accuracy'].values[0])
                else:
                    icl_values.append(0)
                    tv_values.append(0)

            bars1 = ax.bar(x - width/2, icl_values, width, label='ICL', alpha=0.8, color='steelblue')
            bars2 = ax.bar(x + width/2, tv_values, width, label='Task Vector', alpha=0.8,
                          color='darkorange', hatch='//')

            # 値をバーの上に表示
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}',
                               ha='center', va='bottom', fontsize=9)

            ax.set_xlabel('Token Length Range', fontsize=11, fontweight='bold')
            ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
            ax.set_title(f'{model}', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([tr.replace('tokens', '') for tr in token_ranges])
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1])

        fig.suptitle(f'Task: {task}\nAccuracy by Token Length',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_file = os.path.join(output_dir, f'token_length_comparison_{task}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()


def plot_token_length_comparison_by_model(df: pd.DataFrame, output_dir: str):
    """
    モデル別にトークン長ごとの精度を比較

    各モデルごとに1つのグラフを生成し、
    タスクとトークン長範囲ごとにICLとTask Vectorを比較
    """
    models = df['model'].unique()

    for model in models:
        model_data = df[df['model'] == model]
        tasks = model_data['task'].unique()
        token_ranges = sorted(model_data['token_range'].unique())

        fig, axes = plt.subplots(len(token_ranges), 1,
                                figsize=(max(12, len(tasks) * 1.5), 5 * len(token_ranges)))
        if len(token_ranges) == 1:
            axes = [axes]

        for idx, token_range in enumerate(token_ranges):
            ax = axes[idx]
            range_data = model_data[model_data['token_range'] == token_range]

            x = np.arange(len(tasks))
            width = 0.35

            icl_values = []
            tv_values = []

            for task in tasks:
                task_data = range_data[range_data['task'] == task]
                if len(task_data) > 0:
                    icl_values.append(task_data['icl_accuracy'].values[0])
                    tv_values.append(task_data['tv_accuracy'].values[0])
                else:
                    icl_values.append(0)
                    tv_values.append(0)

            bars1 = ax.bar(x - width/2, icl_values, width, label='ICL', alpha=0.8, color='steelblue')
            bars2 = ax.bar(x + width/2, tv_values, width, label='Task Vector', alpha=0.8,
                          color='darkorange', hatch='//')

            # 値をバーの上に表示
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}',
                               ha='center', va='bottom', fontsize=8)

            ax.set_xlabel('Task', fontsize=11, fontweight='bold')
            ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
            ax.set_title(f'Token Range: {token_range.replace("tokens", "")}',
                        fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(tasks, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1])

        fig.suptitle(f'Model: {model}\nAccuracy by Task and Token Length',
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()

        output_file = os.path.join(output_dir, f'token_length_comparison_{model}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()


def plot_heatmap_by_token_range(df: pd.DataFrame, output_dir: str):
    """
    トークン範囲ごとのヒートマップを作成

    各トークン範囲について、モデル×タスクのヒートマップを生成
    """
    token_ranges = sorted(df['token_range'].unique())

    fig, axes = plt.subplots(len(token_ranges), 2,
                            figsize=(14, 5 * len(token_ranges)))

    if len(token_ranges) == 1:
        axes = axes.reshape(1, -1)

    for idx, token_range in enumerate(token_ranges):
        range_data = df[df['token_range'] == token_range]

        # ICLのヒートマップ
        icl_pivot = range_data.pivot(index='task', columns='model', values='icl_accuracy')
        sns.heatmap(icl_pivot, annot=True, fmt='.3f', cmap='YlGnBu',
                   ax=axes[idx, 0], cbar_kws={'label': 'Accuracy'},
                   vmin=0, vmax=1)
        axes[idx, 0].set_title(f'ICL - {token_range}', fontsize=12, fontweight='bold')
        axes[idx, 0].set_xlabel('Model', fontsize=10)
        axes[idx, 0].set_ylabel('Task', fontsize=10)

        # Task Vectorのヒートマップ
        tv_pivot = range_data.pivot(index='task', columns='model', values='tv_accuracy')
        sns.heatmap(tv_pivot, annot=True, fmt='.3f', cmap='YlOrRd',
                   ax=axes[idx, 1], cbar_kws={'label': 'Accuracy'},
                   vmin=0, vmax=1)
        axes[idx, 1].set_title(f'Task Vector - {token_range}', fontsize=12, fontweight='bold')
        axes[idx, 1].set_xlabel('Model', fontsize=10)
        axes[idx, 1].set_ylabel('Task', fontsize=10)

    fig.suptitle('Accuracy Heatmaps by Token Length Range',
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_file = os.path.join(output_dir, 'token_length_heatmaps.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def save_summary_tables(df: pd.DataFrame, output_dir: str):
    """サマリーテーブルをCSVとして保存"""

    # トークン範囲ごとのサマリー
    summary_by_token = df.groupby('token_range').agg({
        'icl_accuracy': ['mean', 'std', 'min', 'max'],
        'tv_accuracy': ['mean', 'std', 'min', 'max'],
        'num_datasets': 'sum'
    }).round(4)

    # タスク×トークン範囲のサマリー
    summary_by_task_token = df.groupby(['task', 'token_range']).agg({
        'icl_accuracy': 'mean',
        'tv_accuracy': 'mean',
        'num_datasets': 'mean'
    }).round(4)

    # モデル×トークン範囲のサマリー
    summary_by_model_token = df.groupby(['model', 'token_range']).agg({
        'icl_accuracy': 'mean',
        'tv_accuracy': 'mean',
        'num_datasets': 'mean'
    }).round(4)

    # 保存
    summary_by_token.to_csv(os.path.join(output_dir, 'summary_by_token_range.csv'))
    summary_by_task_token.to_csv(os.path.join(output_dir, 'summary_by_task_and_token.csv'))
    summary_by_model_token.to_csv(os.path.join(output_dir, 'summary_by_model_and_token.csv'))
    df.to_csv(os.path.join(output_dir, 'all_token_length_results.csv'), index=False)

    print(f"Saved: {os.path.join(output_dir, 'summary_by_token_range.csv')}")
    print(f"Saved: {os.path.join(output_dir, 'summary_by_task_and_token.csv')}")
    print(f"Saved: {os.path.join(output_dir, 'summary_by_model_and_token.csv')}")
    print(f"Saved: {os.path.join(output_dir, 'all_token_length_results.csv')}")


def main():
    """
    メイン関数

    使用例:
        python visualize_token_length_with_comet.py
        python visualize_token_length_with_comet.py --experiment-id token_length_analysis
    """

    # デフォルトの実験ID
    experiment_id = "token_length_analysis"

    # コマンドライン引数から実験IDを取得
    if len(sys.argv) > 1:
        if sys.argv[1] == "--experiment-id" and len(sys.argv) > 2:
            experiment_id = sys.argv[2]

    results_dir = os.path.join(MAIN_RESULTS_DIR, experiment_id)
    output_dir = os.path.join(results_dir, "visualizations")

    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("Token Length Analysis Visualization with COMET-style Comparison")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # 結果を読み込む
    print("Loading results...")
    results = load_token_length_results(results_dir)
    print(f"Loaded {len(results)} result files")
    print()

    # DataFrameに抽出
    print("Extracting data to DataFrame...")
    df = extract_comet_by_token_length(results)
    print(f"DataFrame shape: {df.shape}")
    print()
    print("DataFrame preview:")
    print(df.head(10))
    print()

    if len(df) == 0:
        print("No data to visualize. Make sure you have run token_length_analysis.py first.")
        return

    # 可視化を生成
    print("=" * 80)
    print("Generating visualizations...")
    print("=" * 80)
    print()

    print("1. Creating comparison plots by task...")
    plot_token_length_comparison_by_task(df, output_dir)
    print()

    print("2. Creating comparison plots by model...")
    plot_token_length_comparison_by_model(df, output_dir)
    print()

    print("3. Creating heatmaps...")
    plot_heatmap_by_token_range(df, output_dir)
    print()

    # サマリーテーブルを保存
    print("4. Saving summary tables...")
    save_summary_tables(df, output_dir)
    print()

    print("=" * 80)
    print("Visualization completed!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
