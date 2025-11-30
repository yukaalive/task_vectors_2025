# Bidirectional Averaged Task Vector Experiment

## 概要

en_ja_easyとja_en_easyの2つのタスクベクトルを平均化し、その平均ベクトルをテスト時に使用する実験です。

## 実装内容

### 追加された関数

#### 1. `evaluate_bidirectional_averaged_task_vector()`
- **場所**: `scripts/experiments/main.py` (395行目から)
- **機能**:
  - 2つのタスク（例: en_ja_easy、ja_en_easy）のタスクベクトルを計算
  - タスクベクトルを平均化: `(task1_hiddens + task2_hiddens) / 2.0`
  - 平均ベクトルを使用して各タスクで最適なレイヤーを探索
  - 平均ベクトルを使用してテスト予測を実行
  - ICLと平均TVの性能を比較（COMET、chrF、Accuracyで評価）

#### 2. `run_bidirectional_averaged_experiment()`
- **場所**: `scripts/experiments/main.py` (783行目から)
- **機能**:
  - 実験の実行とresultsの保存を管理
  - 結果は`outputs/results/main/bidirectional_avg/`に保存

### テストスクリプト

#### `test_bidirectional_avg.py`
- 新しい実験を簡単にテストできるスクリプト
- 使用方法:
  ```bash
  # デフォルトモデル（MODELS_TO_EVALUATE[0]）で実行
  python test_bidirectional_avg.py

  # 特定のモデルを指定
  python test_bidirectional_avg.py llama 13B
  ```

## 実験の流れ

1. **タスクベクトルの計算**
   - task1 (en_ja_easy) のdev datasetsからtask1_hiddensを計算
   - task2 (ja_en_easy) のdev datasetsからtask2_hiddensを計算

2. **平均化**
   - `averaged_task_hiddens = (task1_hiddens + task2_hiddens) / 2.0`

3. **最適レイヤーの探索**
   - task1のdev setで全レイヤーをテスト（COMETまたはAccuracy）
   - task2のdev setで全レイヤーをテスト
   - 各タスクで最適なレイヤーを選択

4. **テスト評価**
   - task1のtest setでICLと平均TVを評価
   - task2のtest setでICLと平均TVを評価
   - COMET、chrF、Accuracyを計算

## 出力される結果

結果は以下の形式で保存されます:

```python
{
    "translation_en_ja_easy_and_translation_ja_en_easy": {
        "task1_name": "translation_en_ja_easy",
        "task1_icl_comet": float,
        "task1_avg_tv_comet": float,
        "task1_icl_chrf": float,
        "task1_avg_tv_chrf": float,
        "task1_best_layer": int,
        "task1_dev_accuracy_by_layer": dict,

        "task2_name": "translation_ja_en_easy",
        "task2_icl_comet": float,
        "task2_avg_tv_comet": float,
        "task2_icl_chrf": float,
        "task2_avg_tv_chrf": float,
        "task2_best_layer": int,
        "task2_dev_accuracy_by_layer": dict,

        "task1_prediction_examples": {...},
        "task2_prediction_examples": {...},
        "num_examples": 5,
        "averaged_task_hiddens_shape": tuple,
    }
}
```

## 既存コードへの影響

✅ **既存のコードには一切影響ありません**

- 新しい関数を追加したのみ
- 既存の関数やロジックは変更していません
- run_main_experimentやrun_cross_task_experimentは影響を受けません

## 期待される効果

- en_ja_easy（英→日）のTask Vector性能が低い問題の改善
- 双方向の情報を統合することで、より汎化されたタスク表現の獲得
- ja_en_easy（日→英）の性能維持または向上

## 実行例

```bash
cd /home/yukaalive/2025workspace/task_vectors/21_icl_task_vectors

# テストスクリプトで実行
python test_bidirectional_avg.py

# または、直接main.pyを使用
python -m scripts.experiments.main
```

## 結果の確認

### 方法1: 専用スクリプトで確認（推奨）

```bash
# デフォルトパス（llama_13B）で確認
python view_bidirectional_results.py

# 特定のファイルを指定
python view_bidirectional_results.py outputs/results/main/bidirectional_avg/llama_7B.pkl
```

このスクリプトは以下を表示します：
- タスクベクトル作成に使用したdev examples（最初の3例）
- テスト時に使用したプロンプト（最初の3例）
- 性能メトリクス（COMET、chrF、Retention）
- サンプル予測（最初の3例）

### 方法2: Pythonで直接確認

```python
import pickle

with open('outputs/results/main/bidirectional_avg/llama_13B.pkl', 'rb') as f:
    results = pickle.load(f)

exp_key = 'translation_en_ja_easy_and_translation_ja_en_easy'

# en_ja_easyの結果
print(f"ICL COMET: {results[exp_key]['task1_icl_comet']:.4f}")
print(f"Avg TV COMET: {results[exp_key]['task1_avg_tv_comet']:.4f}")

# ja_en_easyの結果
print(f"ICL COMET: {results[exp_key]['task2_icl_comet']:.4f}")
print(f"Avg TV COMET: {results[exp_key]['task2_avg_tv_comet']:.4f}")

# Dev examples（タスクベクトル作成に使用）
print("\nDev examples used:")
for i, prompt in enumerate(results[exp_key]['task1_dev_examples']['prompts'][:3]):
    print(f"Example {i+1}: {prompt}")

# Test prompts
print("\nTest prompts used:")
for i, prompt in enumerate(results[exp_key]['task1_test_examples']['prompts'][:3]):
    print(f"Example {i+1}: {prompt}")
```
