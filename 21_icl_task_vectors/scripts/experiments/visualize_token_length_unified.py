"""
トークン長分析の統合可視化スクリプト

すべてのメトリクス（Accuracy, chrF, COMET）を1つの図で比較
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
plt.rcParams['font.size'] = 9


def load_all_results(results_dir: str) -> Dict:
    """すべての結果ファイルを読み込む"""
    results = {}

    for pkl_file in Path(results_dir).glob("*.pkl"):
        filename = pkl_file.stem
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            results[filename] = data

    return results


def extract_data_to_dataframe(results: Dict) -> pd.DataFrame:
    """結果をDataFrameに変換

    Note: _single と _easy のサフィックスを削除して統合
    - translation_ja_en_single (0-5tokens) + translation_ja_en_easy (5-10+tokens) → translation_ja_en
    """
    rows = []

    for result_name, data in results.items():
        if 'metadata' not in data or 'categories' not in data:
            continue

        metadata = data['metadata']
        model_name = f"{metadata.get('model_type', '')}_{metadata.get('model_variant', '')}".strip('_')
        task_name = metadata.get('task_name', '')

        # タスク名から _single と _easy を削除して統合
        # これにより、singleの0-5tokensとeasyの5-10+tokensが同じタスクとして扱われる
        task_name = task_name.replace('_single', '').replace('_easy', '')

        # カテゴリごと
        for token_range, cat_data in data['categories'].items():
            row = {
                'model': model_name,
                'task': task_name,
                'token_range': token_range,
                'icl_accuracy': cat_data.get('icl_accuracy', 0),
                'tv_accuracy': cat_data.get('tv_accuracy', 0),
                'icl_chrf': cat_data.get('icl_chrf', 0),
                'tv_chrf': cat_data.get('tv_chrf', 0),
                'icl_comet': cat_data.get('icl_comet', 0),
                'tv_comet': cat_data.get('tv_comet', 0),
                'num_datasets': cat_data.get('num_datasets', 0),
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # chrFが0.1以下の場合、対応するCOMETを0に設定
    df.loc[df['icl_chrf'] <= 0.1, 'icl_comet'] = 0.0
    df.loc[df['tv_chrf'] <= 0.1, 'tv_comet'] = 0.0

    return df


def create_unified_visualization(df: pd.DataFrame, output_dir: str):
    """
    統合された1つの大きな図を作成

    3つのメトリクス（Accuracy, chrF, COMET）× トークン範囲別に表示
    """
    # データの準備
    models = sorted(df['model'].unique())
    tasks = sorted(df['task'].unique())
    token_ranges = sorted(df['token_range'].unique())

    # メトリクスの定義（AccuracyをchrFとCOMETのみに変更）
    metrics = [
        ('chrf', 'chrF'),
        ('comet', 'COMET')
    ]

    # 図のサイズを計算（トークン範囲の数に応じて）
    n_token_ranges = len(token_ranges)
    fig, axes = plt.subplots(n_token_ranges, 2, figsize=(14, 5 * n_token_ranges))

    if n_token_ranges == 1:
        axes = axes.reshape(1, -1)

    # 各トークン範囲 × メトリクスのグリッド
    for row_idx, token_range in enumerate(token_ranges):
        range_data = df[df['token_range'] == token_range]

        for col_idx, (metric_key, metric_name) in enumerate(metrics):
            ax = axes[row_idx, col_idx]

            # データの準備
            x_labels = []
            icl_values = []
            tv_values = []

            for model in models:
                for task in tasks:
                    subset = range_data[(range_data['model'] == model) & (range_data['task'] == task)]
                    if len(subset) > 0:
                        x_labels.append(f"{model}\n{task}")
                        icl_values.append(subset[f'icl_{metric_key}'].values[0])
                        tv_values.append(subset[f'tv_{metric_key}'].values[0])

            if len(x_labels) == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{token_range} - {metric_name}', fontsize=11, fontweight='bold')
                continue

            # バーグラフ
            x = np.arange(len(x_labels))
            width = 0.35

            bars1 = ax.bar(x - width/2, icl_values, width, label='ICL', alpha=0.8, color='steelblue')
            bars2 = ax.bar(x + width/2, tv_values, width, label='Task Vector', alpha=0.8,
                          color='darkorange', hatch='//')

            # 値の表示
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.2f}',
                               ha='center', va='bottom', fontsize=7)

            # 軸の設定
            ax.set_ylabel(metric_name, fontsize=10, fontweight='bold')
            ax.set_title(f'{token_range} - {metric_name}', fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7)

            # 最初の行だけlegendを表示
            if row_idx == 0:
                ax.legend(fontsize=9)

            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1])

    plt.suptitle('Performance by Token Length: chrF and COMET',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_file = os.path.join(output_dir, 'unified_comparison_all_metrics.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def create_all_models_comparison(df: pd.DataFrame, output_dir: str):
    """
    すべてのモデルとタスクを1つのグラフに表示
    横軸: トークン範囲、縦軸: スコア、線: モデル×タスク×方法
    """
    metrics = [
        ('chrf', 'chrF'),
        ('comet', 'COMET')
    ]

    methods = [('icl', 'ICL'), ('tv', 'Task Vector')]

    # トークン範囲の順序を設定
    token_range_order = ['0-5tokens', '5-10tokens', '10-15tokens', '15-20tokens']
    token_ranges = [tr for tr in token_range_order if tr in df['token_range'].values]

    # グラフを2つ作成（chrFとCOMET）
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for metric_idx, (metric_key, metric_name) in enumerate(metrics):
        ax = axes[metric_idx]

        # モデル×タスクの全組み合わせ
        models = sorted(df['model'].unique())
        tasks = sorted(df['task'].unique())

        colors = plt.cm.tab10(np.linspace(0, 1, len(models) * len(tasks) * len(methods)))
        color_idx = 0

        for model in models:
            for task in tasks:
                for method_key, method_name in methods:
                    # データを抽出
                    x_vals = []
                    y_vals = []

                    for token_range in token_ranges:
                        subset = df[
                            (df['model'] == model) &
                            (df['task'] == task) &
                            (df['token_range'] == token_range)
                        ]

                        if len(subset) > 0:
                            x_vals.append(token_range)
                            y_vals.append(subset[f'{method_key}_{metric_key}'].values[0])

                    if len(x_vals) > 0:
                        # ラベルを短縮
                        task_short = task.replace('translation_', '').replace('_easy', '')
                        label = f"{model} - {task_short} ({method_name})"

                        # 線のスタイル
                        linestyle = '-' if method_key == 'icl' else '--'
                        marker = 'o' if method_key == 'icl' else 's'

                        ax.plot(x_vals, y_vals,
                               label=label,
                               marker=marker,
                               linestyle=linestyle,
                               linewidth=2,
                               markersize=6,
                               color=colors[color_idx],
                               alpha=0.7)
                        color_idx += 1

        ax.set_xlabel('Token Range', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_name} by Token Range', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        # x軸のラベルを見やすく、データセット数も追加（各実験の平均）
        dataset_counts = []
        for token_range in token_ranges:
            count = df[df['token_range'] == token_range]['num_datasets'].mean()
            dataset_counts.append(int(count))

        x_labels_with_counts = [f"{tr}\n(n={count})" for tr, count in zip(token_ranges, dataset_counts)]
        ax.set_xticks(range(len(token_ranges)))
        ax.set_xticklabels(x_labels_with_counts, rotation=0)

    plt.suptitle('Performance Comparison: All Models and Token Ranges',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_file = os.path.join(output_dir, 'unified_all_models_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def create_metric_heatmaps(df: pd.DataFrame, output_dir: str):
    """各メトリクスのヒートマップを作成（chrFとCOMETのみ）"""

    metrics = [
        ('chrf', 'chrF'),
        ('comet', 'COMET')
    ]

    methods = [('icl', 'ICL'), ('tv', 'Task Vector')]

    fig, axes = plt.subplots(len(metrics), len(methods), figsize=(14, 5 * len(metrics)))

    for metric_idx, (metric_key, metric_name) in enumerate(metrics):
        for method_idx, (method_key, method_name) in enumerate(methods):
            ax = axes[metric_idx, method_idx]

            # ピボットテーブル作成（タスク×トークン範囲）
            pivot_data = df.pivot_table(
                index='task',
                columns='token_range',
                values=f'{method_key}_{metric_key}',
                aggfunc='mean'
            )

            # ヒートマップ
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd',
                       ax=ax, cbar_kws={'label': metric_name},
                       vmin=0, vmax=1)

            ax.set_title(f'{method_name} - {metric_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Token Range', fontsize=10)
            ax.set_ylabel('Task', fontsize=10)

    plt.suptitle('Heatmaps: chrF and COMET by Token Length',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_file = os.path.join(output_dir, 'heatmaps_all_metrics.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def create_comparison_by_token_range(df: pd.DataFrame, output_dir: str):
    """トークン範囲ごとの比較図（chrFとCOMETのみ）"""

    token_ranges = sorted(df['token_range'].unique())

    for token_range in token_ranges:
        range_data = df[df['token_range'] == token_range]

        if len(range_data) == 0:
            continue

        # 2つのサブプロット
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        metrics = [
            ('chrf', 'chrF'),
            ('comet', 'COMET')
        ]

        for idx, (metric_key, metric_name) in enumerate(metrics):
            ax = axes[idx]

            # データ準備
            tasks = sorted(range_data['task'].unique())
            models = sorted(range_data['model'].unique())

            x = np.arange(len(tasks))
            width = 0.35 / len(models)

            for model_idx, model in enumerate(models):
                model_data = range_data[range_data['model'] == model]

                icl_vals = [model_data[model_data['task'] == task][f'icl_{metric_key}'].values[0]
                           if len(model_data[model_data['task'] == task]) > 0 else 0
                           for task in tasks]
                tv_vals = [model_data[model_data['task'] == task][f'tv_{metric_key}'].values[0]
                          if len(model_data[model_data['task'] == task]) > 0 else 0
                          for task in tasks]

                offset = (model_idx - len(models)/2) * width * 2
                ax.bar(x + offset - width/2, icl_vals, width,
                      label=f'{model} ICL' if idx == 0 else '', alpha=0.8)
                ax.bar(x + offset + width/2, tv_vals, width,
                      label=f'{model} TV' if idx == 0 else '', alpha=0.8, hatch='//')

            ax.set_xlabel('Task', fontsize=11, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
            ax.set_title(metric_name, fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(tasks, rotation=45, ha='right', fontsize=8)
            if idx == 0:
                ax.legend(fontsize=8, ncol=2)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1])

        fig.suptitle(f'chrF and COMET Comparison: {token_range}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        safe_range_name = token_range.replace('-', '_').replace('tokens', '')
        output_file = os.path.join(output_dir, f'comparison_range_{safe_range_name}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()


def create_combined_task_comparison(df: pd.DataFrame, output_dir: str):
    """
    Combine single and easy tasks by token range and create simplified line plots

    Note: Since _single and _easy suffixes are removed in extract_data_to_dataframe,
    tasks are already unified as 'translation_ja_en' and 'translation_en_ja'
    """
    # Translation directions (already unified in the dataframe)
    directions = ['ja_en', 'en_ja']

    # Get all models
    models = sorted(df['model'].unique())

    # Metrics to plot
    metrics = [
        ('chrf', 'chrF'),
        ('comet', 'COMET')
    ]

    # Token range order
    token_ranges = ['0-5tokens', '5-10tokens', '10-15tokens', '15-20tokens']

    # Create one figure per model
    for model in models:
        model_data = df[df['model'] == model]

        # Create figure with 2 subplots (chrF and COMET)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for metric_idx, (metric_key, metric_name) in enumerate(metrics):
            ax = axes[metric_idx]

            # For each direction (ja_en, en_ja)
            for direction in directions:
                # For each method (ICL, Task Vector)
                for method_key, method_name in [('icl', 'ICL'), ('tv', 'Task Vector')]:
                    x_vals = []
                    y_vals = []

                    for token_range in token_ranges:
                        # Use unified task name (no _single or _easy suffix)
                        task_name = f'translation_{direction}'

                        # Extract data for this combination
                        subset = model_data[
                            (model_data['task'] == task_name) &
                            (model_data['token_range'] == token_range)
                        ]

                        if len(subset) > 0:
                            x_vals.append(token_range)
                            y_vals.append(subset[f'{method_key}_{metric_key}'].values[0])

                    if len(x_vals) > 0:
                        # Create label
                        label = f'{direction} ({method_name})'

                        # Line style
                        linestyle = '-' if method_key == 'icl' else '--'
                        marker = 'o' if method_key == 'icl' else 's'

                        # Color by direction
                        color = 'steelblue' if direction == 'ja_en' else 'darkorange'

                        ax.plot(x_vals, y_vals,
                               label=label,
                               marker=marker,
                               linestyle=linestyle,
                               linewidth=2.5,
                               markersize=8,
                               color=color,
                               alpha=0.8 if method_key == 'icl' else 0.6)

            ax.set_xlabel('Token Range', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
            ax.set_title(f'{metric_name} by Token Range', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])
            ax.legend(fontsize=10, loc='best')

            # Add dataset count annotations
            dataset_counts = []
            for token_range in token_ranges:
                count_data = model_data[model_data['token_range'] == token_range]
                if len(count_data) > 0:
                    count = int(count_data['num_datasets'].mean())
                else:
                    count = 0
                dataset_counts.append(count)

            x_labels_with_counts = [f"{tr}\n(n={count})" for tr, count in zip(token_ranges, dataset_counts)]
            ax.set_xticks(range(len(token_ranges)))
            ax.set_xticklabels(x_labels_with_counts, rotation=0)

        plt.suptitle(f'Performance Comparison: {model} (Single for 0-5 tokens, Easy for others)',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save figure
        output_file = os.path.join(output_dir, f'combined_task_comparison_{model}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()


def save_summary_tables(df: pd.DataFrame, output_dir: str):
    """サマリーテーブルをCSVとして保存"""

    # トークン範囲ごとのサマリー
    summary_by_token = df.groupby('token_range').agg({
        'icl_accuracy': ['mean', 'std'],
        'tv_accuracy': ['mean', 'std'],
        'icl_chrf': ['mean', 'std'],
        'tv_chrf': ['mean', 'std'],
        'icl_comet': ['mean', 'std'],
        'tv_comet': ['mean', 'std'],
        'num_datasets': 'sum'
    }).round(4)

    # タスク×トークン範囲
    summary_by_task = df.groupby(['task', 'token_range']).agg({
        'icl_accuracy': 'mean',
        'tv_accuracy': 'mean',
        'icl_chrf': 'mean',
        'tv_chrf': 'mean',
        'icl_comet': 'mean',
        'tv_comet': 'mean',
        'num_datasets': 'mean'
    }).round(4)

    # モデル×トークン範囲
    summary_by_model = df.groupby(['model', 'token_range']).agg({
        'icl_accuracy': 'mean',
        'tv_accuracy': 'mean',
        'icl_chrf': 'mean',
        'tv_chrf': 'mean',
        'icl_comet': 'mean',
        'tv_comet': 'mean',
        'num_datasets': 'mean'
    }).round(4)

    # 保存
    summary_by_token.to_csv(os.path.join(output_dir, 'summary_by_token_range.csv'))
    summary_by_task.to_csv(os.path.join(output_dir, 'summary_by_task.csv'))
    summary_by_model.to_csv(os.path.join(output_dir, 'summary_by_model.csv'))
    df.to_csv(os.path.join(output_dir, 'all_results.csv'), index=False)

    print(f"Saved: {os.path.join(output_dir, 'summary_by_token_range.csv')}")
    print(f"Saved: {os.path.join(output_dir, 'summary_by_task.csv')}")
    print(f"Saved: {os.path.join(output_dir, 'summary_by_model.csv')}")
    print(f"Saved: {os.path.join(output_dir, 'all_results.csv')}")


def main():
    """メイン関数"""

    # デフォルトの実験ID
    experiment_id = "token_length_analysis_v2"

    # コマンドライン引数から実験IDを取得
    if len(sys.argv) > 1:
        if sys.argv[1] == "--experiment-id" and len(sys.argv) > 2:
            experiment_id = sys.argv[2]

    results_dir = os.path.join(MAIN_RESULTS_DIR, experiment_id)
    output_dir = results_dir  # 同じディレクトリに保存

    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return

    print("=" * 80)
    print("Unified Token Length Analysis Visualization")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # 結果を読み込む
    print("Loading results...")
    results = load_all_results(results_dir)
    print(f"Loaded {len(results)} result files")
    print()

    # DataFrameに抽出
    print("Extracting data to DataFrame...")
    df = extract_data_to_dataframe(results)
    print(f"DataFrame shape: {df.shape}")
    print()

    if len(df) == 0:
        print("No data to visualize.")
        return

    print("DataFrame preview:")
    print(df.head(10))
    print()

    # 可視化を生成
    print("=" * 80)
    print("Generating visualizations...")
    print("=" * 80)
    print()

    print("1. Creating all models comparison (line plot)...")
    create_all_models_comparison(df, output_dir)
    print()

    print("2. Creating unified comparison (bar plot by token range)...")
    create_unified_visualization(df, output_dir)
    print()

    print("3. Creating heatmaps...")
    create_metric_heatmaps(df, output_dir)
    print()

    print("4. Creating token range comparisons...")
    create_comparison_by_token_range(df, output_dir)
    print()

    # サマリーテーブルを保存
    print("5. Saving summary tables...")
    save_summary_tables(df, output_dir)
    print()

    # 新しい統合ビジュアライゼーション（single + easy 組み合わせ）
    print("6. Creating combined task comparison (single for 0-5 tokens, easy for others)...")
    create_combined_task_comparison(df, output_dir)
    print()

    print("=" * 80)
    print("Visualization completed!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
