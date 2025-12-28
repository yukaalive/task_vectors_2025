# トークン長分析 V2 - 改善版

## 🎯 主な改善点

1. **自動設定読み込み**: `experiments_config.py` からモデルとタスクを自動取得
2. **包括的メトリクス**: chrF、COMETを計算（Accuracyも内部で計算）
3. **統合可視化**: chrFとCOMETを1つの図で比較
4. **細かいトークン範囲**: 0-5, 5-10, 10-15, 15-20 トークン
5. **既存実装利用**: `core.task_vectors`のタスクベクトル抽出・注入を利用

## 🚀 クイックスタート

### 1. 実行（ワンコマンド）

```bash
cd /home/yukaalive/2025workspace/task_vectors/21_icl_task_vectors/21_icl_task_vectors

# 実行（バックグラウンド）
./run_token_analysis_v2.sh
```

または、conda環境で直接実行：

```bash
conda run -n icl_task_vectors python -u -m scripts.experiments.token_length_analysis_v2
```

### 2. 進捗確認

```bash
# 完了した実験数を確認
ls outputs/results/main/token_length_analysis_v2/*.pkl | wc -l

# ログを監視
tail -f logs/token_length_analysis_v2_*.log
```

### 3. 可視化（実験完了後）

```bash
python -m scripts.experiments.visualize_token_length_unified --experiment-id token_length_analysis_v2
```

## 📊 生成される出力

### 結果ファイル
```
outputs/results/main/token_length_analysis_v2/
├── <model>_<task>.pkl  (各実験の結果)
```

### 可視化ファイル
```
outputs/results/main/token_length_analysis_v2/
├── unified_comparison_all_metrics.png     # 統合比較図（メイン）
├── heatmaps_all_metrics.png               # ヒートマップ
├── comparison_range_*.png                 # トークン範囲別比較
├── summary_by_token_range.csv             # トークン範囲別サマリー
├── summary_by_task.csv                    # タスク別サマリー
├── summary_by_model.csv                   # モデル別サマリー
└── all_results.csv                        # 全結果
```

## 📈 メトリクスの説明

### chrF
- Character-level F-score
- 文字レベルでの類似度を評価
- 0-1の範囲（高いほど良い）
- 完全一致でない翻訳も適切に評価

### COMET
- 翻訳タスクのみ
- ニューラル翻訳評価メトリクス
- 人間の評価との相関が高い
- 0-1の範囲（高いほど良い）

**注**: Accuracyは内部で計算されますが、chrFとCOMETがより適切な評価指標のため、可視化ではこの2つのみを表示します。

## 🔧 カスタマイズ

### トークン範囲の変更

`scripts/experiments/token_length_analysis_v2.py` の `main()` 関数：

```python
run_all_experiments(
    token_ranges=[(0, 5), (5, 10), (10, 15), (15, 20)],  # ← ここを変更
    experiment_id="token_length_analysis_v2"
)
```

### モデルとタスクの変更

`core/experiments_config.py` を編集：

```python
MODELS_TO_EVALUATE = [
    ("swallow", "7B"),
    ("llama", "7B"),
    # 追加・削除可能
]

TASKS_TO_EVALUATE = [
    "translation_ja_en_jesc",
    "translation_en_ja_jesc",
    # 追加・削除可能
]
```

## 📖 使用例

### 例1: デフォルト設定で実行

```bash
./run_token_analysis_v2.sh
```

すべてのモデルとタスクで自動実行されます。

### 例2: カスタム実験ID

```python
# token_length_analysis_v2.py を編集
run_all_experiments(
    token_ranges=[(0, 5), (5, 10), (10, 15), (15, 20)],
    experiment_id="my_custom_experiment"  # ← カスタムID
)
```

```bash
# 可視化時も同じIDを指定
python -m scripts.experiments.visualize_token_length_unified --experiment-id my_custom_experiment
```

## 🎨 可視化の特徴

### unified_comparison_all_metrics.png（メイン図）

- **縦軸**: トークン範囲（0-5, 5-10, 10-15, 15-20）
- **横軸**: 2つのメトリクス（chrF, COMET）
- **バー**: ICL（青）vs Task Vector（オレンジ）
- **比較**: すべてのモデル×タスクの組み合わせ

### heatmaps_all_metrics.png

- **行**: タスク
- **列**: トークン範囲
- **色**: スコア（赤いほど高い）
- **分割**: ICL vs Task Vector

### comparison_range_*.png

- 各トークン範囲ごとの詳細比較
- chrFとCOMETを並べて表示

## ⏱️ 実行時間の目安

- **1実験**: 30-60秒
- **全実験**: `モデル数 × タスク数 × 実験時間`
  - 例: 3モデル × 6タスク = 18実験 → 約15-30分

## 🐛 トラブルシューティング

### 環境エラー

```bash
# conda環境が有効か確認
conda env list

# 環境を明示的に指定
conda run -n icl_task_vectors python -u -m scripts.experiments.token_length_analysis_v2
```

### メモリ不足

```bash
# GPUメモリ確認
nvidia-smi

# 1モデルずつ実行するように experiments_config.py を調整
```

### sacrebleu not found

```bash
# インストール
conda activate icl_task_vectors
pip install sacrebleu
```

## 📚 関連ファイル

- `scripts/experiments/token_length_analysis_v2.py`: メイン実験スクリプト
- `scripts/experiments/visualize_token_length_unified.py`: 統合可視化スクリプト
- `core/experiments_config.py`: モデルとタスクの設定
- `run_token_analysis_v2.sh`: 実行用シェルスクリプト

## ✨ V1からの主な変更点

| 項目 | V1 | V2 |
|------|----|----|
| モデル・タスク | 手動指定 | experiments_config.pyから自動 |
| メトリクス | Accuracyのみ | chrF, COMET（Accuracyも内部計算） |
| 可視化 | 個別グラフ多数 | 統合された図（chrF & COMET） |
| トークン範囲 | 0-10, 10-20, 20-30 | 0-5, 5-10, 10-15, 15-20 |
| タスクベクトル | 既存実装を利用 | 既存実装を利用（変更なし） |

---

**推奨**: V2を使用することをお勧めします！
