translation_ja_en_flow.md                                                                                                                                                        │
│                                                                                                                                                                                  │
│ # 🔄 完全な流れ（具体例：translation_ja_en）                                                                                                                                     │
│                                                                                                                                                                                  │
│ ## 実際の呼び出しフロー                                                                                                                                                          │
│                                                                                                                                                                                  │
│ ### ① main.py:35 - タスクの取得開始                                                                                                                                              │
│                                                                                                                                                                                  │
│ ```python                                                                                                                                                                        │
│ task = get_task_by_name(tokenizer=tokenizer, task_name="translation_ja_en")                                                                                                      │
│ ```                                                                                                                                                                              │
│                                                                                                                                                                                  │
│ ↓                                                                                                                                                                                │
│                                                                                                                                                                                  │
│ ### ② task_helpers.py:156-159 - タスク設定の取得                                                                                                                                 │
│                                                                                                                                                                                  │
│ ```python                                                                                                                                                                        │
│ def get_task_by_name(tokenizer, task_name):                                                                                                                                      │
│     # ① ALL_TASKSから設定を取得                                                                                                                                                  │
│     task_args = ALL_TASKS["translation_ja_en"]                                                                                                                                   │
│     # task_args = {                                                                                                                                                              │
│     #     "task_type": "translation",                                                                                                                                            │
│     #     "task_kwargs": {                                                                                                                                                       │
│     #         "mapping_type": "translation",                                                                                                                                     │
│     #         "mapping_name": "ja_en",                                                                                                                                           │
│     #         "allow_prefix": True                                                                                                                                               │
│     #     }                                                                                                                                                                      │
│     # }                                                                                                                                                                          │
│                                                                                                                                                                                  │
│     # ② get_taskを呼ぶ                                                                                                                                                           │
│     task = get_task("translation", {...}, tokenizer)                                                                                                                             │
│     return task                                                                                                                                                                  │
│ ```                                                                                                                                                                              │
│                                                                                                                                                                                  │
│ ↓                                                                                                                                                                                │
│                                                                                                                                                                                  │
│ ### ③ task_helpers.py:151-153 - タスククラスの取得とインスタンス化                                                                                                               │
│                                                                                                                                                                                  │
│ ```python                                                                                                                                                                        │
│ def get_task(task_type, task_kwargs, tokenizer):                                                                                                                                 │
│     # ③ TASK_TYPE_TO_CLASSから具体的なクラスを取得                                                                                                                               │
│     # TASK_TYPE_TO_CLASS["translation"] → TranslationTask                                                                                                                        │
│                                                                                                                                                                                  │
│     # ④ クラスのインスタンスを作成（ここで__init__が呼ばれる！）                                                                                                                 │
│     task = TranslationTask(                                                                                                                                                      │
│         mapping_type="translation",                                                                                                                                              │
│         mapping_name="ja_en",                                                                                                                                                    │
│         allow_prefix=True,                                                                                                                                                       │
│         tokenizer=tokenizer                                                                                                                                                      │
│     )                                                                                                                                                                            │
│     return task                                                                                                                                                                  │
│ ```                                                                                                                                                                              │
│                                                                                                                                                                                  │
│ ↓                                                                                                                                                                                │
│                                                                                                                                                                                  │
│ ### ④ translation_task.py:22-36 - TranslationTaskの初期化                                                                                                                        │
│                                                                                                                                                                                  │
│ ```python                                                                                                                                                                        │
│ class TranslationTask(MappingTask):                                                                                                                                              │
│     def __init__(self, tokenizer, mapping_type, mapping_name, allow_prefix):                                                                                                     │
│         # ⑤ 親クラスの__init__を呼ぶ                                                                                                                                             │
│         super().__init__(tokenizer, mapping_type, mapping_name, allow_prefix)                                                                                                    │
│         # ⑥ TranslationTask特有の初期化                                                                                                                                          │
│         self.comet_model = None                                                                                                                                                  │
│         self._load_comet_model()                                                                                                                                                 │
│         ...                                                                                                                                                                      │
│ ```                                                                                                                                                                              │
│                                                                                                                                                                                  │
│ ↓                                                                                                                                                                                │
│                                                                                                                                                                                  │
│ ### ⑤ mapping_task.py:22-58 - MappingTaskの初期化                                                                                                                                │
│                                                                                                                                                                                  │
│ ```python                                                                                                                                                                        │
│ class MappingTask(Task):                                                                                                                                                         │
│     def __init__(self, tokenizer, mapping_type, mapping_name, allow_prefix):                                                                                                     │
│         # ⑦ さらに親クラスの__init__を呼ぶ                                                                                                                                       │
│         super().__init__(tokenizer, allow_prefix)                                                                                                                                │
│                                                                                                                                                                                  │
│         # ⑧ JSONファイルを読み込む                                                                                                                                               │
│         mapping_file = "data/translation/ja_en.json"                                                                                                                             │
│         with open(mapping_file) as f:                                                                                                                                            │
│             mapping = json.load(f)  # {"犬": "dog", "猫": "cat", ...}                                                                                                            │
│                                                                                                                                                                                  │
│         self.mapping = mapping                                                                                                                                                   │
│ ```                                                                                                                                                                              │
│                                                                                                                                                                                  │
│ ↓                                                                                                                                                                                │
│                                                                                                                                                                                  │
│ ### ⑥ task.py:10-12 - 基底クラスTaskの初期化                                                                                                                                     │
│                                                                                                                                                                                  │
│ ```python                                                                                                                                                                        │
│ class Task(ABC):                                                                                                                                                                 │
│     def __init__(self, tokenizer, allow_prefix):                                                                                                                                 │
│         self.tokenizer = tokenizer                                                                                                                                               │
│         self.allow_prefix = allow_prefix                                                                                                                                         │
│ ```                                                                                                                                                                              │
│                                                                                                                                                                                  │
│ ### ⑦ インスタンス化完了！                                                                                                                                                       │
│                                                                                                                                                                                  │
│ ```python                                                                                                                                                                        │
│ # taskは TranslationTask のインスタンス                                                                                                                                          │
│ # - tokenizer を持っている                                                                                                                                                       │
│ # - mapping (JSONデータ) を持っている                                                                                                                                            │
│ # - comet_model を持っている                                                                                                                                                     │
│ ```                                                                                                                                                                              │
│                                                                                                                                                                                  │
│ ### ⑧ main.py:38 に戻る                                                                                                                                                          │
│                                                                                                                                                                                  │
│ ```python                                                                                                                                                                        │
│ baseline_datasets = task.create_datasets(num_datasets=100, num_examples=0)                                                                                                       │
│ #                   ↑                                                                                                                                                            │
│ #                   この task は TranslationTask のインスタンス！                                                                                                                │
│ ```                                                                                                                                                                              │
│                                                                                                                                                                                  │
│ ---                                                                                                                                                                              │
│                                                                                                                                                                                  │
│ ## 🎭 ポリモーフィズム（多態性）の仕組み                                                                                                                                         │
│                                                                                                                                                                                  │
│ ### なぜTask型なのにTranslationTaskのメソッドが呼ばれるのか                                                                                                                      │
│                                                                                                                                                                                  │
│ ```python                                                                                                                                                                        │
│ # main.py:35                                                                                                                                                                     │
│ task = get_task_by_name(...)  # 実際は TranslationTask インスタンス                                                                                                              │
│                                                                                                                                                                                  │
│ # main.py:38                                                                                                                                                                     │
│ task.create_datasets(...)                                                                                                                                                        │
│ ```                                                                                                                                                                              │
│                                                                                                                                                                                  │
│ 実行の流れ：                                                                                                                                                                     │
│                                                                                                                                                                                  │
│ 1. **Taskクラスのメソッド実行**                                                                                                                                                  │
│    ↓                                                                                                                                                                             │
│ 2. **self.sample_inputs() を呼ぶ**                                                                                                                                               │
│    ↓                                                                                                                                                                             │
│ 3. **実行時に「このselfは何のインスタンスか？」をチェック**                                                                                                                      │
│    ↓                                                                                                                                                                             │
│ 4. **TranslationTask → MappingTask → sample_inputsを発見！**                                                                                                                     │
│    ↓                                                                                                                                                                             │
│ 5. **MappingTask.sample_inputs が実行される**                                                                                                                                    │
│                                                                                                                                                                                  │
│ ---                                                                                                                                                                              │
│                                                                                                                                                                                  │
│ ## 📋 メソッド解決順序（MRO: Method Resolution Order）                                                                                                                           │
│                                                                                                                                                                                  │
│ Pythonは以下の順番でメソッドを探します：                                                                                                                                         │
│                                                                                                                                                                                  │
│ ### パターン1: `task.create_datasets()` を呼ぶ場合                                                                                                                               │
│                                                                                                                                                                                  │
│ TranslationTask のインスタンスで `task.create_datasets()` を呼ぶと...                                                                                                            │
│                                                                                                                                                                                  │
│ 1. **TranslationTask** に `create_datasets` がある？ → ❌ ない                                                                                                                    │
│ 2. **MappingTask** に `create_datasets` がある？ → ❌ ない                                                                                                                        │
│ 3. **Task** に `create_datasets` がある？ → ✅ **ある！これを実行**                                                                                                               │
│                                                                                                                                                                                  │
│ ### パターン2: Task.create_datasets の中で `self.sample_inputs()` を呼ぶ場合                                                                                                     │
│                                                                                                                                                                                  │
│ 1. **TranslationTask** に `sample_inputs` がある？ → ❌ ない                                                                                                                      │
│ 2. **MappingTask** に `sample_inputs` がある？ → ✅ **ある！これを実行**                                                                                                          │
│                                                                                                                                                                                  │
│ ---                                                                                                                                                                              │
│                                                                                                                                                                                  │
│ ## 📊 クラス階層図                                                                                                                                                               │
│                                                                                                                                                                                  │
│ ```                                                                                                                                                                              │
│ Task (抽象基底クラス)                                                                                                                                                            │
│   ├─ tokenizer                                                                                                                                                                   │
│   ├─ allow_prefix                                                                                                                                                                │
│   └─ create_datasets() メソッド                                                                                                                                                  │
│       │                                                                                                                                                                          │
│       ├─ self.sample_inputs() を呼ぶ                                                                                                                                             │
│       └─ ↓ 実行時に子クラスのメソッドを探す                                                                                                                                      │
│           │                                                                                                                                                                      │
│ MappingTask (Taskを継承)                                                                                                                                                         │
│   ├─ mapping (JSONデータ)                                                                                                                                                        │
│   └─ sample_inputs() メソッド ← ここが実行される！                                                                                                                               │
│       │                                                                                                                                                                          │
│ TranslationTask (MappingTaskを継承)                                                                                                                                              │
│   ├─ comet_model                                                                                                                                                                 │
│   └─ 翻訳特有の機能                                                                                                                                                              │
│ ```                                                                                                                                                                              │
│                                                                                                                                                                                  │
│ ---                                                                                                                                                                              │
│                                                                                                                                                                                  │
│ ## 💡 ポイント                                                                                                                                                                   │
│                                                                                                                                                                                  │
│ - **実行時の型判定**: Pythonは実行時に実際のオブジェクトの型を見てメソッドを探す                                                                                                 │
│ - **継承チェーン**: 子クラス → 親クラス → 祖父クラス の順にメソッドを探す                                                                                                        │
│ - **動的ディスパッチ**: `self.method()` の呼び出しは、実行時の実際のインスタンスの型に基づいて解決される                                                                         │
│                                                                                                                