# ICL Task Vectors UI - Phase 1

## 概要

`scripts/experiments/main.py`の`main()`メソッドの全機能をGradio UIで実現。

## 機能

### ✅ 実装済み機能

1. **モデル選択**
   - 単一または複数モデルの選択
   - `MODELS_TO_EVALUATE`から選択

2. **タスク選択**
   - 単一または複数タスクの選択
   - デフォルトで全タスク選択

3. **実験ID管理**
   - 自動採番: 既存の最大ID + 1
   - 手動指定: カスタムIDを指定可能

4. **実行制御**
   - 実験の開始
   - 実験の停止（途中中断）

5. **リアルタイムログ**
   - 標準出力のキャプチャ
   - 0.5秒ごとの更新
   - 自動スクロール

6. **既存結果のスキップ**
   - 既に実行済みのタスクは自動スキップ
   - `run_main_experiment()`の機能をそのまま継承

## セットアップ

### 1. Gradioのインストール

```bash
pip install gradio
```

### 2. UIの起動

プロジェクトルートから：

```bash
python launch_ui.py
```

または：

```bash
python ui/phase1_main.py
```

### 3. ブラウザでアクセス

```
http://localhost:7860
```

## 使い方

### 基本的な使い方

1. **モデルを選択**
   - 1つ以上のモデルにチェック
   - "Select All Models"で全選択

2. **タスクを選択**
   - デフォルトで全タスク選択済み
   - 必要に応じて変更

3. **実験IDを設定**
   - 自動生成（推奨）: チェックボックスをON
   - 手動指定: チェックボックスをOFFにしてIDを入力

4. **実行**
   - "Run Experiment"ボタンをクリック
   - ログがリアルタイムで表示される

5. **停止**
   - "Stop"ボタンで途中停止可能

### main.pyとの対応

| main.py | UI |
|---------|-----|
| `python main.py` | 全モデルを選択 + Auto-generate ID ON |
| `python main.py 0` | swallow-7Bを選択 |
| `python main.py llama 7B` | llama-7Bを選択 |

### クイックアクション

- **Select All Models**: 全モデルを選択
- **Select All Tasks**: 全タスクを選択
- **Clear Models**: モデル選択をクリア
- **Clear Tasks**: タスク選択をクリア

## 機能詳細

### 実験ID管理

#### 自動生成モード（推奨）
- `get_new_experiment_id()`を使用
- `outputs/results/main/`内の既存IDの最大値 + 1
- 例: 既存のIDが1, 2, 3なら、次は4

#### 手動指定モード
- カスタムIDを指定
- 空欄の場合はデフォルト（IDなし）

### 結果の保存先

```
outputs/results/main/<experiment_id>/
├── swallow_7B.pkl
├── pythia_2.8B.pkl
├── llama_7B.pkl
...
```

### スキップ機能

- 既に存在する結果ファイルは自動的に読み込まれる
- 完了済みのタスクはスキップされる
- ログに"Skipping task..."と表示される

## トラブルシューティング

### UIが起動しない

```bash
# Gradioがインストールされているか確認
pip show gradio

# 再インストール
pip install --upgrade gradio
```

### ポートが使用中

```python
# launch_ui.pyまたはphase1_main.pyを編集
demo.launch(server_port=7861)  # ポート番号を変更
```

### モデルが読み込めない

- 環境変数の確認（LLaMAの場合）
- GPUメモリの確認
- ログでエラーメッセージを確認

## 制限事項

### Phase 1での制限

- ❌ プログレスバー未実装
- ❌ 推定残り時間未表示
- ❌ 結果のグラフ表示なし
- ❌ 過去の実験結果の閲覧機能なし

これらはPhase 2以降で実装予定。

## 次のフェーズ

### Phase 2: 設定の拡張
- パラメータ調整（num_examples, num_datasets）
- 詳細な設定オプション

### Phase 3: プログレス改善
- プログレスバー
- 推定残り時間
- タスク完了数の表示

### Phase 4: 結果表示
- 実験結果の自動表示
- グラフ描画
- 過去の実験との比較

### Phase 5: 高度な機能
- 実験キュー管理
- 設定のプリセット
- 結果のエクスポート

## ファイル構成

```
ui/
├── __init__.py
├── phase1_main.py          # Phase 1のメイン実装
└── README.md               # このファイル

launch_ui.py                # 起動スクリプト
```

## 技術詳細

### ログキャプチャ

- `sys.stdout`/`sys.stderr`をリダイレクト
- `LogCapture`クラスで管理
- 差分のみを返すことで効率化

### スレッド管理

- 実験は別スレッドで実行
- UIスレッドはブロックされない
- 0.5秒ごとにログをポーリング

### 停止機能

- `should_stop`フラグで制御
- スレッドセーフな実装
- モデル間で停止を確認

## 参考

- オリジナル: `scripts/experiments/main.py`
- 設定ファイル: `core/experiments_config.py`
- 結果保存: `scripts/utils.py`
