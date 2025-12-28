# トークン長分析ツール

このツールは、翻訳タスクにおいてトークン長に応じた精度変化を分析するために作成されました。

## 概要

- **既存コードへの影響**: なし（完全に独立して動作）
- **分析対象**: translation データセット
- **トークン長範囲**: 0-5, 5-10, 10-15, 15-20 トークン（データの集中する範囲で細かく分析）
- **評価手法**: ICL（In-Context Learning）と Task Vector

## ファイル構成

- `token_length_analysis.py`: メインの分析スクリプト
- `visualize_token_length_analysis.py`: 結果の可視化スクリプト
- `TOKEN_LENGTH_ANALYSIS_README.md`: このファイル

## 使用方法

### 1. トークン長分析の実行

基本的な使い方:

```bash
python scripts/experiments/token_length_analysis.py <model_type> <model_variant> <task_name> [task_name2 ...]
```

例:

```bash
# 単一タスクの実行
python scripts/experiments/token_length_analysis.py gemma 2b translation_ja_en_jesc

# 複数タスクの実行
python scripts/experiments/token_length_analysis.py gemma 2b translation_ja_en_jesc translation_en_ja_jesc

# 日本語-英語の全タスクを実行
python scripts/experiments/token_length_analysis.py gemma 2b translation_ja_en_jesc translation_en_ja_jesc translation_ja_en_easy translation_en_ja_easy
```

### 2. 結果の可視化

分析が完了したら、結果を可視化できます:

```bash
# 特定の結果ファイルを可視化
python scripts/experiments/visualize_token_length_analysis.py outputs/results/main/token_length_analysis/gemma_2b_translation_ja_en_jesc.pkl

# 実験IDのすべての結果を一括可視化
python scripts/experiments/visualize_token_length_analysis.py --experiment-id token_length_analysis
```

## 出力

### 分析結果ファイル

結果は以下のディレクトリに保存されます:

```
outputs/results/main/token_length_analysis/
├── gemma_2b_translation_ja_en_jesc.pkl
├── gemma_2b_translation_en_ja_jesc.pkl
└── ...
```

各 `.pkl` ファイルには以下の情報が含まれます:

- トークン長カテゴリごとのデータセット数
- 各カテゴリでの ICL 精度
- 各カテゴリでの Task Vector 精度
- 全体の精度
- メタデータ（モデル情報、タスク名、実行時間など）

### 可視化結果

可視化スクリプトは2種類のグラフを生成します:

1. **`*_accuracy.png`**: トークン長ごとの精度比較とデータセット分布
2. **`*_difference.png`**: ICLとTask Vectorの精度差

## 利用可能なタスク

### 日本語-英語（JESCデータセット）
- `translation_ja_en_jesc`: 日本語→英語
- `translation_en_ja_jesc`: 英語→日本語

### 日本語-英語（Easyデータセット）
- `translation_ja_en_easy`: 日本語→英語
- `translation_en_ja_easy`: 英語→日本語

### 日本語-英語（シングルトークン）
- `translation_ja_en_single`: 日本語→英語
- `translation_en_ja_single`: 英語→日本語

### その他の言語ペア（シングルトークン）
- `translation_fr_en`: フランス語→英語
- `translation_en_fr`: 英語→フランス語
- `translation_it_en`: イタリア語→英語
- `translation_en_it`: 英語→イタリア語
- `translation_es_en`: スペイン語→英語
- `translation_en_es`: 英語→スペイン語

## トークン長範囲のカスタマイズ

デフォルトでは `[(0, 10), (10, 20), (20, 30)]` の範囲で分析しますが、コード内の `token_ranges` パラメータを変更することでカスタマイズ可能です。

例えば、より細かい範囲で分析したい場合:

```python
# token_length_analysis.py の main() 関数内で
token_ranges=[(0, 5), (5, 10), (10, 15), (15, 20), (20, 30)]
```

## 技術的な詳細

### 動作の仕組み

1. **データセット作成**: 指定されたタスクのテストデータセットを作成
2. **トークン数カウント**: 各データセットの `test_input` のトークン数をカウント
3. **カテゴリ分け**: トークン数に基づいてデータセットを分類
4. **予測生成**: ICLとTask Vectorで予測を生成
5. **精度計算**: カテゴリごとに精度を計算
6. **結果保存**: pickle形式で結果を保存

### 既存コードとの関係

- **完全に独立**: 既存の `main.py` を一切変更していません
- **共有モジュール**: `core` パッケージの関数を利用していますが、読み取りのみ
- **互換性**: 既存の実験と同時に実行可能

### パフォーマンス

- データセット数: テスト50、開発50（`main.py` と同じ設定）
- GPU利用: 既存コードと同じGPU設定を使用
- 実行時間: タスクとモデルのサイズに依存（通常数分～数十分）

## トラブルシューティング

### よくある問題

1. **GPU メモリ不足**
   - より小さいモデルで試す
   - データセット数を減らす（コード内の `num_test_datasets` を変更）

2. **結果が既に存在する**
   - スクリプトは既存の結果をスキップします
   - 再実行したい場合は、既存の `.pkl` ファイルを削除してください

3. **可視化エラー**
   - matplotlib がインストールされていることを確認
   - 結果ファイルが正しく生成されているか確認

## 結果の解釈

### 精度グラフの読み方

- **ICL（青）**: 従来の In-Context Learning の精度
- **Task Vector（オレンジ）**: Task Vector 手法の精度
- 各バーの高さが精度を表します（0.0～1.0）

### データセット分布の重要性

- トークン長カテゴリによってデータセット数が偏る可能性があります
- データセット数が少ないカテゴリの結果は統計的に不安定な場合があります
- 右側のグラフでデータセット分布を確認してください

### 精度差グラフの読み方

- **緑のバー**: Task Vector が ICL より優れている
- **赤のバー**: ICL が Task Vector より優れている
- バーの高さが精度差の大きさを表します

## 今後の拡張

このツールは以下のように拡張可能です:

- より細かいトークン長範囲での分析
- 他のタスクタイプ（algorithmic, linguistic, knowledge）への対応
- COMET スコアのトークン長別分析
- 複数モデルの比較可視化

## サポート

問題が発生した場合は、以下を確認してください:

1. 既存の実験が正常に動作するか（`main.py`）
2. 必要な依存パッケージがインストールされているか
3. データセットファイルが正しい場所にあるか

---

**注意**: このツールは既存のコードを一切変更しません。既存の実験と並行して安全に使用できます。
