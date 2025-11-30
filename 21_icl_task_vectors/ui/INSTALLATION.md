# UI セットアップガイド

## 前提条件

このUIは既存のICL Task Vectorsプロジェクトの環境を使用します。

### 必須環境

既に以下がインストールされている必要があります：
- Python 3.10+
- PyTorch
- Transformers
- その他の依存関係（`requirements.in`参照）

## インストール手順

### ステップ1: プロジェクトの依存関係を確認

既存の環境を使用する場合：

```bash
# Conda環境をアクティベート
conda activate icl_task_vectors
```

新しい環境の場合：

```bash
# Conda環境を作成
conda env create -f environment.yml -n icl_task_vectors
conda activate icl_task_vectors
```

### ステップ2: Gradioをインストール

```bash
pip install gradio
```

### ステップ3: UIの動作確認

```bash
# プロジェクトルートで実行
python launch_ui.py
```

または：

```bash
python ui/phase1_main.py
```

### ステップ4: ブラウザでアクセス

```
http://localhost:7860
```

## トラブルシューティング

### エラー: `ModuleNotFoundError: No module named 'gradio'`

**解決策**:
```bash
pip install gradio
```

### エラー: `ModuleNotFoundError: No module named 'transformers'`

**解決策**: プロジェクトの依存関係をインストール
```bash
pip install transformers torch accelerate
```

### エラー: `ModuleNotFoundError: No module named 'dotenv'`

**解決策**:
```bash
pip install python-dotenv
```

### ポート7860が使用中

**解決策1**: 別のポートを使用

`launch_ui.py`または`ui/phase1_main.py`を編集：

```python
demo.launch(
    server_name="0.0.0.0",
    server_port=7861,  # ポート番号を変更
    share=False,
    show_error=True
)
```

**解決策2**: 既存のプロセスを停止

```bash
# ポート7860を使用しているプロセスを確認
lsof -i :7860

# プロセスを停止
kill -9 <PID>
```

### UIが起動しない

1. Pythonのバージョンを確認：
```bash
python --version  # 3.10以上が必要
```

2. 依存関係を再インストール：
```bash
pip install --upgrade gradio
```

3. ログを確認：
```bash
python launch_ui.py 2>&1 | tee ui_launch.log
```

## 完全セットアップ（クリーンインストール）

全ての依存関係を一から インストールする場合：

```bash
# 1. Conda環境を作成
conda env create -f environment.yml -n icl_task_vectors

# 2. 環境をアクティベート
conda activate icl_task_vectors

# 3. requirements.inから追加パッケージをインストール
pip install gradio

# 4. .envファイルを設定（必要に応じて）
cp .env_example .env
# .envを編集してLLAMA_DIRなどを設定

# 5. UIを起動
python launch_ui.py
```

## 動作確認

UIが正常に起動したら、以下を確認：

1. **ブラウザでアクセス**
   - http://localhost:7860 が開く

2. **モデルリストが表示される**
   - 7つのモデルが選択可能

3. **タスクリストが表示される**
   - 5つの翻訳タスクが選択可能

4. **設定が変更できる**
   - モデル・タスクの選択
   - 実験IDの設定

5. **実験を実行できる**（軽量テスト）
   - 1つのモデルと1つのタスクを選択
   - "Run Experiment"をクリック
   - ログが表示される

## 最小限のテスト実行

実際に実験を実行せずにUIの動作を確認：

```bash
# テストスクリプトを実行（依存関係のチェックのみ）
python ui/test_ui.py
```

成功すれば以下のような出力：

```
============================================================
Test Summary
============================================================
✅ PASS: Imports
✅ PASS: LogCapture
✅ PASS: Experiment ID
✅ PASS: Configuration
✅ PASS: Paths
✅ PASS: UI Creation

6/6 tests passed

🎉 All tests passed! You can now run the UI:
   python launch_ui.py
```

## 次のステップ

UIが正常に起動したら：

1. **軽量テスト**: 小さいモデル（pythia-2.8B）で1タスクを実行
2. **本番実行**: 必要なモデルとタスクで実験
3. **結果確認**: `outputs/results/main/`で結果をチェック

## サポート

問題が解決しない場合：

1. `ui/README.md` を参照
2. GitHubのIssuesを確認
3. ログファイルを確認（`logs/`ディレクトリ）
