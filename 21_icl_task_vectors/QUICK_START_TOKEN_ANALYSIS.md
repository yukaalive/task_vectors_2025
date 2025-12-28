# トークン長分析 - クイックスタートガイド

experiments_config.pyに定義されているすべてのモデルとタスクでトークン長分析を実行する方法です。

## 現在の設定

### モデル (4個)
1. swallow 7B
2. llama 7B
3. youko 8B
4. llama 13B

### タスク (2個)
1. translation_ja_en_easy (日本語→英語)
2. translation_en_ja_easy (英語→日本語)

**合計: 4モデル × 2タスク = 8個の実験**

## 実行方法

### 方法1: 全自動実行（推奨）

すべてのモデルとタスクの組み合わせを自動で実行します：

```bash
cd /home/yukaalive/2025workspace/task_vectors/21_icl_task_vectors/21_icl_task_vectors

# バックグラウンドで実行
./run_all_token_length_analysis.sh

# または、フォアグラウンドで実行（進捗が見える）
python run_all_token_length_analysis.py
```

### 方法2: 個別実行

特定のモデル×タスクだけ実行したい場合：

```bash
cd /home/yukaalive/2025workspace/task_vectors/21_icl_task_vectors/21_icl_task_vectors

# 例1: swallow 7Bで日本語→英語タスク
./run_token_length_analysis.sh swallow 7B translation_ja_en_easy

# 例2: llama 7Bで両タスク
./run_token_length_analysis.sh llama 7B translation_ja_en_easy translation_en_ja_easy

# 例3: Pythonモジュールとして直接実行
python -m scripts.experiments.token_length_analysis swallow 7B translation_ja_en_easy
```

## 進捗確認

### バックグラウンド実行の場合

```bash
# ログファイルをリアルタイム監視
tail -f logs/token_length_analysis_all_*.log

# プロセスの確認
ps aux | grep token_length_analysis

# プロセスの停止（必要な場合）
kill <PID>
```

### 結果ファイルの確認

```bash
# 結果ファイルの一覧
ls -lh outputs/results/main/token_length_analysis/

# 完了した実験数を確認
ls outputs/results/main/token_length_analysis/*.pkl | wc -l
```

## 結果の可視化

すべての実験が完了したら、結果を可視化します：

```bash
# すべての結果を一括可視化
python -m scripts.experiments.visualize_token_length_analysis --experiment-id token_length_analysis

# 生成されるグラフの確認
ls outputs/results/main/token_length_analysis/*.png
```

各実験ごとに2つのグラフが生成されます：
- `*_accuracy.png`: トークン長ごとの精度比較とデータセット分布
- `*_difference.png`: ICLとTask Vectorの精度差

## 実行時間の目安

1つの実験（モデル×タスク）あたり：
- 小型モデル（7B）: 約10-20分
- 大型モデル（13B）: 約20-40分

**全実験の合計**: 約2-4時間（並列実行なしの場合）

## 出力ファイル

```
outputs/results/main/token_length_analysis/
├── swallow_7B_translation_ja_en_easy.pkl         # 結果データ
├── swallow_7B_translation_ja_en_easy_accuracy.png    # 精度グラフ
├── swallow_7B_translation_ja_en_easy_difference.png  # 精度差グラフ
├── swallow_7B_translation_en_ja_easy.pkl
├── swallow_7B_translation_en_ja_easy_accuracy.png
├── swallow_7B_translation_en_ja_easy_difference.png
├── llama_7B_translation_ja_en_easy.pkl
├── ... (他のモデル×タスクの組み合わせ)
```

## トラブルシューティング

### GPU メモリ不足

```bash
# GPUメモリの確認
nvidia-smi

# 小さいモデルから順に実行
./run_token_length_analysis.sh swallow 7B translation_ja_en_easy
# 完了を待ってから次のモデル
./run_token_length_analysis.sh llama 7B translation_ja_en_easy
```

### 途中で中断した場合

- 既に完了した実験はスキップされます（結果ファイルが既に存在する場合）
- 再度実行すれば、未完了の実験のみが実行されます

### 再実行したい場合

```bash
# 特定の結果ファイルを削除
rm outputs/results/main/token_length_analysis/swallow_7B_translation_ja_en_easy.pkl

# すべての結果を削除して最初から
rm -rf outputs/results/main/token_length_analysis/
```

## 設定のカスタマイズ

### トークン長範囲について

デフォルトでは `[(0, 5), (5, 10), (10, 15), (15, 20)]` の範囲で分析します。
これは、データセットのほとんどが0-20トークンの範囲に集中しているため、その範囲を細かく分析できるように設定されています。

カスタマイズしたい場合は `run_all_token_length_analysis.py` の以下の行を編集：

```python
token_ranges=[(0, 5), (5, 10), (10, 15), (15, 20)]  # デフォルト
# ↓ 例: より広い範囲
token_ranges=[(0, 10), (10, 20), (20, 30)]
```

### 実行するモデル・タスクの変更

`core/experiments_config.py` を編集：

```python
TASKS_TO_EVALUATE = [
    "translation_ja_en_easy",
    "translation_en_ja_easy",
    # "translation_ja_en_jesc",  # コメント解除で追加
]

MODELS_TO_EVALUATE = [
    ("swallow", "7B"),
    ("llama", "7B"),
    # ("llama", "13B"),  # コメントアウトで除外
]
```

## 次のステップ

1. **実験実行**: `./run_all_token_length_analysis.sh` で全実験を開始
2. **進捗監視**: `tail -f logs/token_length_analysis_all_*.log`
3. **結果確認**: 完了したら結果ファイルを確認
4. **可視化**: `python -m scripts.experiments.visualize_token_length_analysis --experiment-id token_length_analysis`
5. **分析**: 生成されたグラフを見てトークン長による精度変化を分析

## 詳細情報

より詳しい情報は以下を参照：
- `TOKEN_LENGTH_ANALYSIS_README.md`: 詳細なドキュメント
- `scripts/experiments/token_length_analysis.py`: メインスクリプト
- `scripts/experiments/visualize_token_length_analysis.py`: 可視化スクリプト

---

**注意**: この分析は既存のコードを変更しません。main.pyの実験と並行して実行可能です。
