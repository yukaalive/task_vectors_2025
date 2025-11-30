# タスクベクトルの動作説明 - 簡単なベクトル例

## 概要

`modulated_generate`関数とその関連関数の動作を、非常に簡単なベクトルの例を使って説明します。

## 前提となる設定

```
モデル: 3層のTransformer (embedding層 + layer 0, 1, 2)
隠れ層の次元: 4次元
テストデータ数: 2個
```

---

## 1. `get_single_context_task_hiddens` の動作

### 目的
特定のテスト入力に依存しない「タスクの本質」を捉えるベクトルを抽出する

### 入力
- `datasets`: 2個のFewShotDataset
- `num_test_inputs_to_avg`: 2

### プロセス

#### ステップ1: 代替テスト入力で新しいデータセットを作成

```python
# 元のdatasets (2個)
datasets[0]: train=[例1, 例2], test="X"
datasets[1]: train=[例3, 例4], test="Y"

# new_datasetsを作成 (2個 × 2 = 4個)
new_datasets[0]: train=[例1, 例2], test="A"  # dataset[0]用の代替test
new_datasets[1]: train=[例1, 例2], test="B"  # dataset[0]用の代替test
new_datasets[2]: train=[例3, 例4], test="C"  # dataset[1]用の代替test
new_datasets[3]: train=[例3, 例4], test="D"  # dataset[1]用の代替test
```

**理由**: 元のテスト入力"X"や"Y"に依存しない、タスク全般に適用できるベクトルを得るため

#### ステップ2: モデルの順伝播で隠れ層を取得

```python
# traced_forwardの出力
forward_trace.residual_stream.hidden.shape = (4, 4, seq_len, 4)
# (バッチ4, 層4(embedding+3層), シーケンス長, 隠れ次元4)

# 最後のトークンのみ抽出
task_hiddens = hidden[:, :, -1, :]  # shape: (4, 4, 4)
```

#### ステップ3: 平均化して安定化

具体的な数値例:

```python
# layer 0 の4次元全て
task_hiddens_pre[0, 0, :] = [1.2, 0.8, 0.5, 0.3]  # new_dataset[0]
task_hiddens_pre[1, 0, :] = [1.3, 0.7, 0.6, 0.4]  # new_dataset[1]
task_hiddens_pre[2, 0, :] = [2.1, 1.5, 1.8, 1.2]  # new_dataset[2]
task_hiddens_pre[3, 0, :] = [2.2, 1.4, 1.9, 1.1]  # new_dataset[3]

# 平均化: 2つずつグループ化して平均
task_hiddens_post = task_hiddens_pre.view(2, 2, 4, 4).mean(dim=1)
# shape: (2, 4, 4) = (datasets数, 層数, 隠れ次元)

# 平均後の値
task_hiddens_post[0, 0, :] = [(1.2+1.3)/2, (0.8+0.7)/2, (0.5+0.6)/2, (0.3+0.4)/2]
                            = [1.25, 0.75, 0.55, 0.35]

task_hiddens_post[1, 0, :] = [(2.1+2.2)/2, (1.5+1.4)/2, (1.8+1.9)/2, (1.2+1.1)/2]
                            = [2.15, 1.45, 1.85, 1.15]

# embedding層を除去
task_hiddens = task_hiddens_post[:, 1:]  # shape: (2, 3, 4)
```

### 最終出力
`task_hiddens.shape = (2, 3, 4)` = (データセット数, レイヤー数, 隠れ次元)

---

## 2. `modulated_forward` の動作

### 目的
モデルの特定の層にタスクベクトルを注入して順伝播を実行する

### 入力
- `task_hiddens`: (2, 3, 4) のテンソル
- `intermediate_layer`: 1 (layer 1に注入)

### プロセス

#### ステップ1: 注入パラメータの準備

```python
# intermediate_layerを各バッチ用に複製
intermediate_layer = torch.tensor([1, 1])  # 両方のデータセットでlayer 1

# injection_positionsを-1に設定(最後の位置)
injection_positions = torch.tensor([-1, -1])

# 注入する隠れ層を選択
task_hiddens_to_inject = task_hiddens[range(2), intermediate_layer]
# = task_hiddens[:, 1]  # layer 1の隠れ層を選択
# shape: (2, 4)

# 具体例:
task_hiddens_to_inject[0] = [0.5, 0.3, 0.8, 0.2]  # dataset 0用
task_hiddens_to_inject[1] = [0.6, 0.4, 0.7, 0.3]  # dataset 1用
```

#### ステップ2: HiddenInjectorの動作

```python
# layer 1でのフック発動
def inject_hidden(mod, inp, out):
    hidden_states = out[0]  # shape: (2, seq_len, 4)

    # 例: seq_len=10, 最後の位置(-1)に注入
    hidden_states[0, -1, :] = [0.5, 0.3, 0.8, 0.2]  # dataset 0
    hidden_states[1, -1, :] = [0.6, 0.4, 0.7, 0.3]  # dataset 1

    return out
```

### 順伝播の流れ

```
入力 → embedding層
     → layer 0 (通常処理)
     → layer 1 (最後の位置にtask_vectorを注入★)
     → layer 2 (注入された値を使って処理)
     → 出力logits
```

---

## 3. `modulated_generate` の動作

### 目的
タスクベクトルを注入して新しいトークンを生成する

### ステップ1: 最初のトークン生成 (タスクベクトル注入あり)

```python
# modulated_forwardで最初の順伝播
first_forward_outputs.logits.shape = (2, seq_len, vocab_size)

# 最後の位置のlogitsからトークンを予測
logits_last = first_forward_outputs.logits[:, -1, :]  # shape: (2, vocab_size)

# 例: vocab_size=100
logits_last[0] = [0.1, 0.3, ..., 2.5, ..., 0.2]  # 最大値が位置42
logits_last[1] = [0.2, 0.1, ..., 1.8, ..., 0.4]  # 最大値が位置73

# argmaxで最も確率が高いトークンを選択
first_predicted_token_ids = logits_last.argmax(dim=-1)
# = [42, 73]  # shape: (2,)
# → unsqueeze → [[42], [73]]  # shape: (2, 1)
```

### ステップ2: 残りのトークン生成 (通常のgenerate、max_new_tokens > 1の場合)

```python
# 最初に生成したトークンを入力に結合
full_input_ids = torch.cat([inputs["input_ids"], first_predicted_token_ids], dim=-1)
# 元の入力: (2, 10)
# 新トークン: (2, 1)
# 結合後: (2, 11)

# model.generateで残りを生成 (max_new_tokens=3なら、あと2トークン)
output_ids = model.generate(
    input_ids=full_input_ids,
    max_new_tokens=2,  # 3 - 1 = 2
    past_key_values=updated_past_key_values,
)
# output_ids.shape = (2, 13)  # 11 + 2

# 元の入力部分を除去
new_ids = output_ids[:, 10:]  # shape: (2, 3)
# 例: [[42, 15, 28],
#      [73, 91, 12]]
```

### ステップ3: デコード

```python
# tokenizer.batch_decodeで文字列に変換
predictions = decode_predictions(new_ids, tokenizer)
# = ["cat", "dog"]  # 例
```

---

## 全体の流れ (数値例)

```
【入力】
test_datasets = [dataset_0, dataset_1]

↓ get_single_context_task_hiddens

【タスクベクトル抽出】
task_hiddens[0] = [[1.25, 0.75, ...], [0.5, 0.3, ...], [0.8, 0.2, ...]]  # 3層分
task_hiddens[1] = [[2.15, 1.45, ...], [0.6, 0.4, ...], [0.9, 0.1, ...]]

↓ 最適レイヤー選択 (例: layer 1)

↓ modulated_generate

【ステップ1】modulated_forward
- layer 1の最後の位置にtask_hiddens[:, 1]を注入
- 注入されたベクトル:
  dataset_0: [0.5, 0.3, 0.8, 0.2]
  dataset_1: [0.6, 0.4, 0.7, 0.3]

【ステップ2】最初のトークン生成
- logitsから最大値選択: [42, 73]

【ステップ3】残りのトークン生成 (通常のgenerate)
- 最終出力: [[42, 15, 28], [73, 91, 12]]

↓ decode_predictions

【予測結果】
predictions = ["cat", "dog"]
```

---

## 重要なポイント

### 1. タスクベクトルの本質
複数の異なるテスト入力で平均化することで、特定の入力に依存しない「タスクの本質」を捉えるベクトルを得る

### 2. 注入位置
最後のトークン位置(-1)に注入することで、次のトークン予測に直接影響を与える

### 3. レイヤー選択
dev setで各レイヤーの精度を測定し、最も性能が良いレイヤーを選択する

### 4. 2段階生成
- **第1段階**: タスクベクトルを注入して最初の1トークンを生成
- **第2段階**: 通常のgenerateで残りのトークンを生成 (max_new_tokens > 1の場合)

### 5. 利点
この仕組みにより、Few-shotの例を直接プロンプトに含めなくても、タスクの「コンテキスト」をモデルに与えることができる

---

## 図解: タスクベクトル注入の仕組み

```
通常のForward:
入力 → [Embed] → [Layer0] → [Layer1] → [Layer2] → 出力

タスクベクトル注入:
入力 → [Embed] → [Layer0] → [Layer1 + タスクベクトル注入] → [Layer2] → 出力
                                      ↑
                              最後の位置に注入
```

---

## 関連ファイル

- メイン実装: `core/task_vectors.py:234-299` (modulated_generate)
- タスクベクトル抽出: `core/task_vectors.py:130-190` (get_single_context_task_hiddens)
- 注入機構: `core/models/context_managers/forward_modifiers/hidden_injector.py:7-55` (HiddenInjector)
- 順伝播処理: `core/models/utils/inference.py:20-56` (traced_forward, modified_forward)
