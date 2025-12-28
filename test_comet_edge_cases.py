#!/usr/bin/env python3
"""
Test COMET score for various edge cases
"""
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['COMET_QUIET'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import logging
logging.getLogger("comet").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)

from comet import download_model, load_from_checkpoint

def test_edge_cases():
    import sys
    from io import StringIO

    print("Loading COMET model...")
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    try:
        model_path = download_model("Unbabel/wmt22-comet-da")
        model = load_from_checkpoint(model_path)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    print("Model loaded!\n")

    # Test cases
    test_cases = [
        {"name": "空文字列", "prediction": ""},
        {"name": "ピリオドのみ", "prediction": "."},
        {"name": "記号のみ", "prediction": "!!!"},
        {"name": "数字のみ", "prediction": "123"},
        {"name": "ランダムな文字", "prediction": "asdfghjkl"},
        {"name": "同じ文字の繰り返し", "prediction": "aaaaaaaaaa"},
        {"name": "正解", "prediction": "His wife opened the door for him."},
        {"name": "参照と全く同じ（完全一致）", "prediction": "His wife opened the door for him."},
    ]

    source = "彼の妻は彼のためにドアを開けた。"
    reference = "His wife opened the door for him."

    print("=" * 70)
    print(f"Source:    {source}")
    print(f"Reference: {reference}")
    print("=" * 70)
    print()

    for test_case in test_cases:
        data = [{
            "src": source,
            "mt": test_case["prediction"],
            "ref": reference
        }]

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        try:
            result = model.predict(data, batch_size=1, gpus=1, progress_bar=False)
            score = result.scores[0]
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        print(f"{test_case['name']:30s} | Pred: '{test_case['prediction']:30s}' | COMET: {score:.4f}")

if __name__ == "__main__":
    test_edge_cases()
