#!/usr/bin/env python3
"""
Test COMET score for empty string
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

def test_empty_string():
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
        {"name": "スペース1つ", "prediction": " "},
        {"name": "スペース複数", "prediction": "   "},
        {"name": "改行のみ", "prediction": "\n"},
        {"name": "タブのみ", "prediction": "\t"},
    ]

    source = "彼の妻は彼のためにドアを開けた。"
    reference = "His wife opened the door for him."

    for test_case in test_cases:
        data = [{
            "src": source,
            "mt": test_case["prediction"],
            "ref": reference
        }]

        print(f"Testing: {test_case['name']}")
        print(f"Prediction: '{test_case['prediction']}'")
        print(f"Length: {len(test_case['prediction'])}")

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

        print(f"COMET Score: {score:.4f}")
        print("-" * 50)
        print()

if __name__ == "__main__":
    test_empty_string()
