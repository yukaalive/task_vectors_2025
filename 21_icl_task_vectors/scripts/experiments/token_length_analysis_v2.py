"""
トークン長に応じた精度変化を分析するスクリプト（改善版）

- experiments_config.pyからモデルとタスクを自動取得
- COMET、chrF、Accuracyを計算
- トークン長別に詳細な分析
- 事前計算されたベストレイヤーを使用して高速化
"""

from dotenv import load_dotenv
load_dotenv(".env")
import sys
import os
import pickle
import time
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import warnings

from transformers import PreTrainedModel, PreTrainedTokenizer
import sacrebleu

from scripts.utils import MAIN_RESULTS_DIR
from core.data.task_helpers import get_task_by_name
from core.data.datasets.few_shot_dataset import FewShotDataset
from core.models.llm_loading import load_model_and_tokenizer
from core.task_vectors import run_icl, run_task_vector
from core.utils.misc import limit_gpus, seed_everything
from core.experiments_config import MODELS_TO_EVALUATE, TASKS_TO_EVALUATE

warnings.filterwarnings("ignore", category=UserWarning)

# 事前計算されたベストレイヤーのディレクトリ
BEST_LAYERS_DIR = "/home/yukaalive/2025workspace/task_vectors/22_icl_task_vectors_merged/outputs/results/main/best_layers_baseline"


def load_best_layers(model_type: str, model_variant: str) -> Optional[Dict[str, int]]:
    """事前計算されたベストレイヤー情報をロード

    Returns:
        タスク名 -> ベストレイヤーのマッピング、またはNone（ファイルが見つからない場合）
    """
    pkl_file = Path(BEST_LAYERS_DIR) / f"{model_type}_{model_variant}.pkl"

    if not pkl_file.exists():
        print(f"Warning: Best layers file not found: {pkl_file}")
        return None

    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        best_layers = {}
        for task_name, task_data in data.items():
            if 'tv_dev_accruacy_by_layer' in task_data:
                layer_accuracies = task_data['tv_dev_accruacy_by_layer']
                best_layer = max(layer_accuracies.keys(), key=lambda k: layer_accuracies[k])
                best_layers[task_name] = best_layer

        return best_layers
    except Exception as e:
        print(f"Error loading best layers from {pkl_file}: {e}")
        return None


def count_tokens(text: str, tokenizer: PreTrainedTokenizer) -> int:
    """テキストのトークン数をカウント"""
    return len(tokenizer.encode(text, add_special_tokens=False))


def categorize_datasets_by_token_length(
    datasets: List[FewShotDataset],
    tokenizer: PreTrainedTokenizer,
    ranges: List[Tuple[int, int]] = [(0, 5), (5, 10), (10, 15), (15, 20)],
    max_datasets_per_range: int = 50
) -> Dict[str, List[int]]:
    """データセットをtest_inputのトークン長でカテゴリ分け（インデックスベース）

    各トークン範囲で最大max_datasets_per_range個まで
    """
    categorized = {f"{min_t}-{max_t}tokens": [] for min_t, max_t in ranges}

    for idx, dataset in enumerate(datasets):
        token_count = count_tokens(dataset.test_input, tokenizer)

        for min_t, max_t in ranges:
            if min_t <= token_count < max_t:
                range_key = f"{min_t}-{max_t}tokens"
                # 各範囲で最大50個まで
                if len(categorized[range_key]) < max_datasets_per_range:
                    categorized[range_key].append(idx)
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


def calculate_chrf_score(
    predictions: List[str],
    references: List[str]
) -> float:
    """chrFスコアを計算"""
    if len(predictions) == 0 or len(references) == 0:
        return 0.0

    try:
        # sacrebleuでchrFを計算
        chrf = sacrebleu.corpus_chrf(predictions, [references])
        return chrf.score / 100.0  # 0-1の範囲に正規化
    except Exception as e:
        print(f"Warning: chrF calculation failed: {e}")
        return 0.0


def calculate_comet_score(
    task,
    sources: List[str],
    predictions: List[str],
    references: List[str]
) -> float:
    """COMETスコアを計算（タスクがCOMETをサポートしている場合）"""
    if len(predictions) == 0 or len(references) == 0:
        return 0.0

    # タスクがevaluate_with_cometメソッドを持っているか確認
    if not hasattr(task, 'evaluate_with_comet'):
        return 0.0

    try:
        result = task.evaluate_with_comet(sources, predictions, references)
        return result.get('comet', 0.0)
    except Exception as e:
        print(f"Warning: COMET calculation failed: {e}")
        return 0.0


def evaluate_task_by_token_length(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task_name: str,
    num_examples: int,
    token_ranges: List[Tuple[int, int]] = [(0, 5), (5, 10), (10, 15), (15, 20)],
    best_layer: Optional[int] = None
) -> Dict:
    """タスクをトークン長ごとに評価（COMET、chrF、Accuracyを含む）

    Args:
        best_layer: 事前計算されたベストレイヤー。指定された場合、レイヤー選択をスキップ
    """
    seed_everything(41)

    task = get_task_by_name(tokenizer=tokenizer, task_name=task_name)
    generation_mode = "single" if "_single" in task_name else "multi"

    print(f"Generation mode: {generation_mode}")
    print(f"Token ranges: {token_ranges}")

    # データセット作成 - 各トークン範囲から50件ずつ確保するため、多めに作成
    num_datasets_per_range = 50
    num_initial_datasets = 1000  # 各範囲から50件確保するため、十分な数を作成

    print(f"Creating {num_initial_datasets} initial test datasets...")
    initial_test_datasets = task.create_datasets(num_datasets=num_initial_datasets, num_examples=num_examples)
    initial_dev_datasets = task.create_datasets(num_datasets=num_initial_datasets, num_examples=num_examples)

    # トークン長でカテゴリ分け
    print("Categorizing datasets by token length...")
    categorized_test = categorize_datasets_by_token_length(initial_test_datasets, tokenizer, token_ranges, max_datasets_per_range=num_datasets_per_range)
    categorized_dev = categorize_datasets_by_token_length(initial_dev_datasets, tokenizer, token_ranges, max_datasets_per_range=num_datasets_per_range)

    # 各範囲から選択されたデータセットのみを使用
    test_datasets = []
    test_indices_map = {}  # 元のインデックスから新しいインデックスへのマッピング
    current_idx = 0

    for category in sorted(categorized_test.keys()):
        indices = categorized_test[category]
        test_indices_map[category] = list(range(current_idx, current_idx + len(indices)))
        for idx in indices:
            test_datasets.append(initial_test_datasets[idx])
        current_idx += len(indices)

    dev_datasets = []
    for category in sorted(categorized_dev.keys()):
        for idx in categorized_dev[category]:
            dev_datasets.append(initial_dev_datasets[idx])

    print("\nDataset distribution by token length:")
    for category in sorted(categorized_test.keys()):
        test_count = len(categorized_test[category])
        dev_count = len(categorized_dev[category])
        print(f"  {category}: {test_count} test datasets, {dev_count} dev datasets")

    results = {
        "token_ranges": token_ranges,
        "dataset_counts": {cat: len(indices) for cat, indices in categorized_test.items()},
        "categories": {}
    }

    # ICLとTask Vectorの予測
    print("\n===========↓Regular ICL↓===========")
    icl_predictions = run_icl(model, tokenizer, task, test_datasets, generation_mode=generation_mode)
    print("===========↑Regular ICL↑===========\n")

    print("===========↓Task Vectors↓===========")
    max_new_tokens = 1 if generation_mode == "single" else 30

    # ベストレイヤーが指定されている場合は、そのレイヤーのみをテスト
    layers_to_test = None
    if best_layer is not None:
        layers_to_test = [best_layer]
        print(f"使用するベストレイヤー: {best_layer} (事前計算済み)")

    tv_predictions, tv_dev_accuracy_by_layer, task_hiddens = run_task_vector(
        model,
        tokenizer,
        task,
        test_datasets,
        dev_datasets,
        layers_to_test=layers_to_test,
        generation_mode=generation_mode,
        max_new_tokens=max_new_tokens,
    )
    print("===========↑Task Vectors↑===========\n")

    # カテゴリごとに各種スコアを計算
    print("Calculating metrics by token length category...")
    for category in sorted(categorized_test.keys()):
        new_indices = test_indices_map[category]

        if len(new_indices) == 0:
            print(f"  {category}: No datasets, skipping")
            continue

        # 対応する予測とデータセットを取得（新しいインデックスを使用）
        category_icl_predictions = [icl_predictions[i] for i in new_indices]
        category_tv_predictions = [tv_predictions[i] for i in new_indices]
        category_datasets = [test_datasets[i] for i in new_indices]

        # 参照データ
        category_sources = [ds.test_input for ds in category_datasets]
        category_references = [ds.test_output for ds in category_datasets]

        # Accuracy計算
        icl_accuracy = calculate_accuracy_for_datasets(task, category_icl_predictions, category_datasets)
        tv_accuracy = calculate_accuracy_for_datasets(task, category_tv_predictions, category_datasets)

        # chrFスコア計算
        icl_chrf = calculate_chrf_score(category_icl_predictions, category_references)
        tv_chrf = calculate_chrf_score(category_tv_predictions, category_references)

        # COMETスコア計算（翻訳タスクのみ）
        icl_comet = 0.0
        tv_comet = 0.0
        if "translation" in task_name:
            # chrFが0の場合、COMETも0にする
            if icl_chrf > 0.001:
                icl_comet = calculate_comet_score(task, category_sources, category_icl_predictions, category_references)
            if tv_chrf > 0.001:
                tv_comet = calculate_comet_score(task, category_sources, category_tv_predictions, category_references)

        print(f"  {category}:")
        print(f"    ICL - Accuracy: {icl_accuracy:.4f}, chrF: {icl_chrf:.4f}, COMET: {icl_comet:.4f}")
        print(f"    TV  - Accuracy: {tv_accuracy:.4f}, chrF: {tv_chrf:.4f}, COMET: {tv_comet:.4f}")

        results["categories"][category] = {
            "icl_accuracy": icl_accuracy,
            "tv_accuracy": tv_accuracy,
            "icl_chrf": icl_chrf,
            "tv_chrf": tv_chrf,
            "icl_comet": icl_comet,
            "tv_comet": tv_comet,
            "num_datasets": len(category_datasets)
        }

    # 全体のスコアも計算
    all_sources = [ds.test_input for ds in test_datasets]
    all_references = [ds.test_output for ds in test_datasets]

    overall_icl_accuracy = calculate_accuracy_for_datasets(task, icl_predictions, test_datasets)
    overall_tv_accuracy = calculate_accuracy_for_datasets(task, tv_predictions, test_datasets)
    overall_icl_chrf = calculate_chrf_score(icl_predictions, all_references)
    overall_tv_chrf = calculate_chrf_score(tv_predictions, all_references)

    overall_icl_comet = 0.0
    overall_tv_comet = 0.0
    if "translation" in task_name:
        # chrFが0の場合、COMETも0にする
        if overall_icl_chrf > 0.001:
            overall_icl_comet = calculate_comet_score(task, all_sources, icl_predictions, all_references)
        if overall_tv_chrf > 0.001:
            overall_tv_comet = calculate_comet_score(task, all_sources, tv_predictions, all_references)

    results["overall"] = {
        "icl_accuracy": overall_icl_accuracy,
        "tv_accuracy": overall_tv_accuracy,
        "icl_chrf": overall_icl_chrf,
        "tv_chrf": overall_tv_chrf,
        "icl_comet": overall_icl_comet,
        "tv_comet": overall_tv_comet,
        "num_datasets": len(test_datasets)
    }

    print(f"\nOverall Metrics:")
    print(f"  ICL - Accuracy: {overall_icl_accuracy:.4f}, chrF: {overall_icl_chrf:.4f}, COMET: {overall_icl_comet:.4f}")
    print(f"  TV  - Accuracy: {overall_tv_accuracy:.4f}, chrF: {overall_tv_chrf:.4f}, COMET: {overall_tv_comet:.4f}")

    return results


def run_all_experiments(
    experiment_id: str = "token_length_analysis_v2"
):
    """experiments_config.pyのすべてのモデルとタスクで実験を実行"""

    # Override tasks to include easy and single translation tasks
    # single: 0-5 tokens only
    # easy: 5-10, 10-15, 15-20 tokens
    tasks_config = {
        "translation_ja_en_easy": [(5, 10), (10, 15), (15, 20)],
        "translation_en_ja_easy": [(5, 10), (10, 15), (15, 20)],
        "translation_ja_en_single": [(0, 5)],
        "translation_en_ja_single": [(0, 5)],
    }

    results_dir = os.path.join(MAIN_RESULTS_DIR, experiment_id)
    os.makedirs(results_dir, exist_ok=True)

    limit_gpus(range(0, 8))  # Same as main.py

    print("=" * 80)
    print("Token Length Analysis - Comprehensive Version")
    print("=" * 80)
    print(f"\nModels to evaluate ({len(MODELS_TO_EVALUATE)}):")
    for i, (model_type, model_variant) in enumerate(MODELS_TO_EVALUATE, 1):
        print(f"  {i}. {model_type} {model_variant}")

    print(f"\nTasks to evaluate ({len(tasks_config)}):")
    for i, (task_name, token_ranges) in enumerate(tasks_config.items(), 1):
        print(f"  {i}. {task_name} -> {token_ranges}")

    total_experiments = len(MODELS_TO_EVALUATE) * len(tasks_config)
    print(f"\nTotal experiments to run: {total_experiments}")
    print("=" * 80)
    print()

    experiment_count = 0
    for model_idx, (model_type, model_variant) in enumerate(MODELS_TO_EVALUATE, 1):
        print()
        print("#" * 80)
        print(f"# Model {model_idx}/{len(MODELS_TO_EVALUATE)}: {model_type} {model_variant}")
        print("#" * 80)
        print()

        # モデルを一度だけロード
        print(f"Loading model: {model_type} {model_variant}...")
        model, tokenizer = load_model_and_tokenizer(model_type, model_variant)
        print("Model loaded successfully.\n")

        # 事前計算されたベストレイヤーをロード
        print(f"Loading best layers for {model_type} {model_variant}...")
        best_layers = load_best_layers(model_type, model_variant)
        if best_layers:
            print(f"Loaded {len(best_layers)} best layers from baseline")
            for task, layer in best_layers.items():
                print(f"  {task}: Layer {layer}")
        else:
            print("No pre-computed best layers found, will compute on the fly")
        print()

        for task_idx, (task_name, token_ranges) in enumerate(tasks_config.items(), 1):
            experiment_count += 1

            results_file = os.path.join(results_dir, f"{model_type}_{model_variant}_{task_name}.pkl")

            # 既存の結果があればスキップ
            if os.path.exists(results_file):
                print(f"[{experiment_count}/{total_experiments}] Skipping: {model_type} {model_variant} - {task_name} (already exists)")
                continue

            print()
            print("=" * 80)
            print(f"Experiment {experiment_count}/{total_experiments}")
            print(f"Model: {model_type} {model_variant}")
            print(f"Task: {task_name}")
            print(f"Token ranges: {token_ranges}")
            print("=" * 80)
            print()

            try:
                tic = time.time()

                # ベストレイヤーを取得（存在する場合）
                best_layer = None
                if best_layers and task_name in best_layers:
                    best_layer = best_layers[task_name]

                results = evaluate_task_by_token_length(
                    model,
                    tokenizer,
                    task_name,
                    num_examples=5,
                    token_ranges=token_ranges,
                    best_layer=best_layer
                )
                elapsed = time.time() - tic

                results["metadata"] = {
                    "model_type": model_type,
                    "model_variant": model_variant,
                    "task_name": task_name,
                    "num_examples": 5,
                    "elapsed_time": elapsed
                }

                # 結果を保存
                with open(results_file, "wb") as f:
                    pickle.dump(results, f)

                print()
                print(f"✓ Completed: {model_type} {model_variant} - {task_name} ({elapsed:.2f}s)")
                print()

            except Exception as e:
                print()
                print(f"✗ Error in {model_type} {model_variant} - {task_name}:")
                print(f"  {e}")
                import traceback
                traceback.print_exc()
                print()
                print("Continuing with next experiment...")
                print()

    print()
    print("=" * 80)
    print("All experiments completed!")
    print("=" * 80)
    print()
    print("To visualize results, run:")
    print(f"  python -m scripts.experiments.visualize_token_length_unified --experiment-id {experiment_id}")
    print()


def main():
    """メイン関数"""
    run_all_experiments(
        experiment_id="token_length_analysis_v2"
    )


if __name__ == "__main__":
    main()
