"""
トークン長に応じた精度変化を分析するスクリプト

既存のmain.pyを一切変更せず、独立して動作します。
データセットをトークン長（0-10, 10-20, 20-30）で分類し、各カテゴリでの精度を計測します。
"""

from dotenv import load_dotenv
load_dotenv(".env")
import sys
import os
import pickle
import time
from typing import Optional, Dict, List, Tuple

from transformers import PreTrainedModel, PreTrainedTokenizer

from scripts.utils import MAIN_RESULTS_DIR
from core.data.task_helpers import get_task_by_name
from core.data.datasets.few_shot_dataset import FewShotDataset
from core.models.llm_loading import load_model_and_tokenizer
from core.task_vectors import run_icl, run_task_vector
from core.utils.misc import limit_gpus, seed_everything
from core.experiments_config import MODELS_TO_EVALUATE


def count_tokens(text: str, tokenizer: PreTrainedTokenizer) -> int:
    """テキストのトークン数をカウント"""
    return len(tokenizer.encode(text, add_special_tokens=False))


def categorize_datasets_by_token_length(
    datasets: List[FewShotDataset],
    tokenizer: PreTrainedTokenizer,
    ranges: List[Tuple[int, int]] = [(0, 5), (5, 10), (10, 15), (15, 20)]
) -> Dict[str, List[int]]:
    """
    データセットをtest_inputのトークン長でカテゴリ分け

    Args:
        datasets: データセットのリスト
        tokenizer: トークナイザー
        ranges: トークン長の範囲のリスト [(min, max), ...]

    Returns:
        カテゴリ名をキー、データセットのインデックスリストを値とする辞書
    """
    categorized = {f"{min_t}-{max_t}tokens": [] for min_t, max_t in ranges}

    for idx, dataset in enumerate(datasets):
        token_count = count_tokens(dataset.test_input, tokenizer)

        for min_t, max_t in ranges:
            if min_t <= token_count < max_t:
                categorized[f"{min_t}-{max_t}tokens"].append(idx)
                break

    return categorized


def calculate_accuracy_for_datasets(
    task,
    predictions: List[str],
    datasets: List[FewShotDataset]
) -> float:
    """データセットのリストに対して精度を計算"""
    if len(datasets) == 0:
        return 0.0

    correct = 0
    for pred, dataset in zip(predictions, datasets):
        if task.compare_outputs(pred, dataset.test_output):
            correct += 1

    return correct / len(datasets)


def evaluate_task_by_token_length(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task_name: str,
    num_examples: int,
    token_ranges: List[Tuple[int, int]] = [(0, 5), (5, 10), (10, 15), (15, 20)]
) -> Dict:
    """
    タスクをトークン長ごとに評価

    Args:
        model: 評価するモデル
        tokenizer: トークナイザー
        task_name: タスク名
        num_examples: Few-shotの例の数
        token_ranges: トークン長の範囲

    Returns:
        各トークン長カテゴリでの結果を含む辞書
    """
    seed_everything(41)

    task = get_task_by_name(tokenizer=tokenizer, task_name=task_name)

    # 生成モードの決定
    generation_mode = "single" if "_single" in task_name else "multi"
    print(f"Generation mode: {generation_mode}")

    # テストデータセットの作成
    num_test_datasets = 50
    num_dev_datasets = 50

    print(f"Creating {num_test_datasets} test datasets...")
    test_datasets = task.create_datasets(num_datasets=num_test_datasets, num_examples=num_examples)
    dev_datasets = task.create_datasets(num_datasets=num_dev_datasets, num_examples=num_examples)

    # トークン長でカテゴリ分け
    print("Categorizing datasets by token length...")
    categorized_test = categorize_datasets_by_token_length(test_datasets, tokenizer, token_ranges)

    # 各カテゴリのデータセット数を表示
    print("\nDataset distribution by token length:")
    for category, indices in categorized_test.items():
        print(f"  {category}: {len(indices)} datasets")

    results = {
        "token_ranges": token_ranges,
        "dataset_counts": {cat: len(indices) for cat, indices in categorized_test.items()},
        "categories": {}
    }

    # ICLの予測
    print("\n===========↓Regular ICL↓===========")
    icl_predictions = run_icl(model, tokenizer, task, test_datasets, generation_mode=generation_mode)
    print("===========↑Regular ICL↑===========\n")

    # Task Vectorの予測
    print("===========↓Task Vectors↓===========")
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
    print("===========↑Task Vectors↑===========\n")

    # カテゴリごとに精度を計算
    print("Calculating accuracy by token length category...")
    for category, category_indices in categorized_test.items():
        if len(category_indices) == 0:
            print(f"  {category}: No datasets, skipping")
            continue

        # 対応する予測とデータセットを取得
        category_icl_predictions = [icl_predictions[i] for i in category_indices]
        category_tv_predictions = [tv_predictions[i] for i in category_indices]
        category_datasets = [test_datasets[i] for i in category_indices]

        # 精度を計算
        icl_accuracy = calculate_accuracy_for_datasets(task, category_icl_predictions, category_datasets)
        tv_accuracy = calculate_accuracy_for_datasets(task, category_tv_predictions, category_datasets)

        print(f"  {category}:")
        print(f"    ICL Accuracy: {icl_accuracy:.4f}")
        print(f"    TV Accuracy:  {tv_accuracy:.4f}")

        results["categories"][category] = {
            "icl_accuracy": icl_accuracy,
            "tv_accuracy": tv_accuracy,
            "num_datasets": len(category_datasets)
        }

    # 全体の精度も計算
    overall_icl_accuracy = calculate_accuracy_for_datasets(task, icl_predictions, test_datasets)
    overall_tv_accuracy = calculate_accuracy_for_datasets(task, tv_predictions, test_datasets)

    results["overall"] = {
        "icl_accuracy": overall_icl_accuracy,
        "tv_accuracy": overall_tv_accuracy,
        "num_datasets": len(test_datasets)
    }

    print(f"\nOverall Accuracy:")
    print(f"  ICL: {overall_icl_accuracy:.4f}")
    print(f"  TV:  {overall_tv_accuracy:.4f}")

    return results


def run_token_length_analysis(
    model_type: str,
    model_variant: str,
    task_name: str,
    num_examples: int = 5,
    token_ranges: List[Tuple[int, int]] = [(0, 5), (5, 10), (10, 15), (15, 20)],
    experiment_id: str = "token_length_analysis"
) -> None:
    """
    トークン長分析を実行

    Args:
        model_type: モデルのタイプ
        model_variant: モデルのバリアント
        task_name: 評価するタスク名
        num_examples: Few-shotの例の数
        token_ranges: トークン長の範囲
        experiment_id: 実験ID
    """
    # 結果保存先
    results_dir = os.path.join(MAIN_RESULTS_DIR, experiment_id)
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{model_type}_{model_variant}_{task_name}.pkl")

    # 既存の結果があればスキップ
    if os.path.exists(results_file):
        print(f"Results already exist at {results_file}, skipping...")
        with open(results_file, "rb") as f:
            results = pickle.load(f)
        print("Loaded existing results:")
        print_results_summary(results)
        return

    limit_gpus(range(0, 8))

    print(f"Loading model: {model_type} {model_variant}...")
    model, tokenizer = load_model_and_tokenizer(model_type, model_variant)
    print("Model loaded successfully.\n")

    print("=" * 80)
    print(f"Task: {task_name}")
    print(f"Token ranges: {token_ranges}")
    print("=" * 80)

    tic = time.time()
    results = evaluate_task_by_token_length(
        model,
        tokenizer,
        task_name,
        num_examples,
        token_ranges
    )
    elapsed = time.time() - tic

    results["metadata"] = {
        "model_type": model_type,
        "model_variant": model_variant,
        "task_name": task_name,
        "num_examples": num_examples,
        "elapsed_time": elapsed
    }

    # 結果を保存
    with open(results_file, "wb") as f:
        pickle.dump(results, f)

    print(f"\n{'=' * 80}")
    print(f"Analysis completed in {elapsed:.2f} seconds")
    print(f"Results saved to: {results_file}")
    print("=" * 80)


def print_results_summary(results: Dict) -> None:
    """結果のサマリーを表示"""
    print("\nResults Summary:")
    print("-" * 80)

    if "metadata" in results:
        meta = results["metadata"]
        print(f"Model: {meta.get('model_type')} {meta.get('model_variant')}")
        print(f"Task: {meta.get('task_name')}")
        print(f"Elapsed time: {meta.get('elapsed_time', 0):.2f}s")
        print()

    print("Dataset distribution:")
    for category, count in results["dataset_counts"].items():
        print(f"  {category}: {count} datasets")
    print()

    print("Accuracy by token length:")
    for category, cat_results in results["categories"].items():
        print(f"  {category}:")
        print(f"    ICL: {cat_results['icl_accuracy']:.4f}")
        print(f"    TV:  {cat_results['tv_accuracy']:.4f}")
    print()

    if "overall" in results:
        overall = results["overall"]
        print(f"Overall accuracy:")
        print(f"  ICL: {overall['icl_accuracy']:.4f}")
        print(f"  TV:  {overall['tv_accuracy']:.4f}")

    print("-" * 80)


def main():
    """
    メイン関数

    使用例:
        # 単一タスクを実行
        python token_length_analysis.py gemma 2b translation_ja_en_jesc

        # 複数のタスクを実行
        python token_length_analysis.py gemma 2b translation_ja_en_jesc translation_en_ja_jesc
    """
    if len(sys.argv) < 4:
        print("Usage: python token_length_analysis.py <model_type> <model_variant> <task_name> [task_name2 ...]")
        print("\nExample:")
        print("  python token_length_analysis.py gemma 2b translation_ja_en_jesc")
        print("  python token_length_analysis.py gemma 2b translation_ja_en_jesc translation_en_ja_jesc")
        print("\nAvailable tasks:")
        print("  - translation_ja_en_jesc")
        print("  - translation_en_ja_jesc")
        print("  - translation_ja_en_easy")
        print("  - translation_en_ja_easy")
        print("  - translation_ja_en_single")
        print("  - translation_en_ja_single")
        print("  - translation_fr_en")
        print("  - translation_en_fr")
        print("  - translation_it_en")
        print("  - translation_en_it")
        print("  - translation_es_en")
        print("  - translation_en_es")
        sys.exit(1)

    model_type = sys.argv[1]
    model_variant = sys.argv[2]
    task_names = sys.argv[3:]

    print(f"Model: {model_type} {model_variant}")
    print(f"Tasks to evaluate: {task_names}")
    print()

    for i, task_name in enumerate(task_names, 1):
        print(f"\n{'#' * 80}")
        print(f"# Task {i}/{len(task_names)}: {task_name}")
        print(f"{'#' * 80}\n")

        run_token_length_analysis(
            model_type=model_type,
            model_variant=model_variant,
            task_name=task_name,
            num_examples=5,
            token_ranges=[(0, 10), (10, 20), (20, 30)]
        )


if __name__ == "__main__":
    main()
