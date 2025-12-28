#!/usr/bin/env python3
"""
Simple script to test COMET and chrF scores for translation examples.
Usage: python test_comet_chrf_score.py
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
import sacrebleu
from langdetect import detect, LangDetectException


# Global COMET model (load once)
_comet_model = None

def get_comet_model():
    """Load COMET model once and reuse"""
    global _comet_model
    if _comet_model is None:
        import sys
        from io import StringIO
        # Suppress all stdout and stderr during model loading
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        try:
            model_path = download_model("Unbabel/wmt22-comet-da")
            _comet_model = load_from_checkpoint(model_path)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    return _comet_model


def evaluate_translation(source, prediction, reference, target_lang="en"):
    """
    Evaluate translation using multiple metrics with language validation.

    Args:
        source: Source language text (Japanese)
        prediction: Model's translation (English)
        reference: Ground truth translation (English)
        target_lang: Expected target language code (default: "en")

    Returns:
        dict: Dictionary containing all scores and validation results
    """
    results = {
        'source': source,
        'prediction': prediction,
        'reference': reference
    }

    # Step 1: Language validation
    try:
        if not prediction or prediction.strip() == "":
            results['detected_language'] = 'empty'
            results['language_valid'] = False
            results['warning'] = 'Empty prediction'
        else:
            pred_lang = detect(prediction)
            results['detected_language'] = pred_lang
            results['language_valid'] = (pred_lang == target_lang)
    except LangDetectException:
        results['detected_language'] = 'unknown'
        results['language_valid'] = False
        results['warning'] = 'Could not detect language'

    # If wrong language, set chrF scores to zero but continue to calculate COMET
    if not results['language_valid']:
        results['chrf'] = 0.0
        results['error'] = f"Language mismatch: expected {target_lang}, got {results['detected_language']}"
        # Continue to calculate COMET to observe its behavior
    else:
        # Step 2: Calculate chrF (character n-grams only)
        chrf_result = sacrebleu.sentence_chrf(
            prediction,
            [reference],
            word_order=0,  # Character n-grams only
            char_order=6   # Default: up to 6-grams
        )
        results['chrf'] = chrf_result.score / 100.0

    # Step 4: Calculate COMET (always calculate, even for wrong language)
    import sys
    from io import StringIO

    model = get_comet_model()
    data = [{
        "src": source,
        "mt": prediction,
        "ref": reference
    }]

    # Suppress stdout and stderr during prediction
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    try:
        comet_result = model.predict(data, batch_size=1, gpus=1, progress_bar=False)
        results['comet'] = comet_result.scores[0]
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    return results


def format_score_display(results):
    """
    Format the evaluation results for display.
    
    Args:
        results: Dictionary containing evaluation scores
    
    Returns:
        str: Formatted string for display
    """
    output = []
    
    # Language validation
    if results['language_valid']:
        output.append(f"✓ Language: {results['detected_language']} (Valid)")
    else:
        output.append(f"✗ Language: {results['detected_language']} (Invalid - Expected: en)")
        if 'error' in results:
            output.append(f"   {results['error']}")
    
    # Scores (always display, even if zero)
    output.append("\n--- Scores ---")
    
    # chrF
    chrf = results['chrf']
    output.append(f"chrF:       {chrf:.4f}")
    
    # COMET (always display actual score)
    comet = results['comet']
    output.append(f"COMET:      {comet:.4f}")
    
    # Warning if exists
    if 'warning' in results:
        output.append(f"\n⚠️  WARNING: {results['warning']}")
    
    return "\n".join(output)


def main():
    """Test COMET and chrF scores on predefined examples"""

    print("=" * 80)
    print("Translation Evaluation Tool (COMET + chrF)")
    print("=" * 80)
    print("\nEvaluates translations using:")
    print("  • Language validation (langdetect)")
    print("  • chrF (character n-gram F-score)")
    print("  • COMET (neural metric)")
    print("\n⚠️  COMET Limitation:")
    print("  • Empty/meaningless outputs still receive ~40% score")
    print("  • This is a known baseline bias issue")
    print("  • Use chrF + language detection for robust evaluation")
    print("\nLoading COMET model...")

    # Load model once at the start
    get_comet_model()
    print("Model loaded successfully!\n")

    examples = [
        {
            "name": "✓ 正解の例",
            "source": "彼の妻は彼のためにドアを開けた。",
            "prediction": "His wife opened the door for him.",
            "reference": "His wife opened the door for him."
        },
        {
            "name": "✓ ほとんど正解の例（同義語使用）",
            "source": "彼の妻は彼のためにドアを開けた。",
            "prediction": "His spouse opened the door for him.",
            "reference": "His wife opened the door for him."
        },
        {
            "name": "✗ 文法エラー + 内容の誤り",
            "source": "彼の妻は彼のためにドアを開けた。",
            "prediction": "His wif opened the door for cats.",
            "reference": "His wife opened the door for him."
        },
        {
            "name": "✗ 翻訳なし（元の日本語のまま）",
            "source": "彼の妻は彼のためにドアを開けた。",
            "prediction": "彼の妻は彼のためにドアを開けた。",
            "reference": "His wife opened the door for him."
        },
        {
            "name": "✗ 部分翻訳（途中まで）",
            "source": "彼の妻は彼のためにドアを開けた。",
            "prediction": "His wife は彼のために door opened.",
            "reference": "His wife opened the door for him."
        },
        {
            "name": "✗ 単語のみ（不完全な翻訳）",
            "source": "彼の妻は彼のためにドアを開けた。",
            "prediction": "wife door",
            "reference": "His wife opened the door for him."
        },
        {
            "name": "✗ 完全に関係ない文章",
            "source": "彼の妻は彼のためにドアを開けた。",
            "prediction": "I like to eat sushi and ramen.",
            "reference": "His wife opened the door for him."
        },
        {
            "name": "✗ 空文字列",
            "source": "彼の妻は彼のためにドアを開けた。",
            "prediction": "",
            "reference": "His wife opened the door for him."
        },
        {
            "name": "✗ 意味は合っているが文法が完全に崩壊",
            "source": "彼の妻は彼のためにドアを開けた。",
            "prediction": "door open wife he for",
            "reference": "His wife opened the door for him."
        },
        {
            "name": "✗ 主語と目的語の混同",
            "source": "彼の妻は彼のためにドアを開けた。",
            "prediction": "He opened the door for his wife.",
            "reference": "His wife opened the door for him."
        },
        {
            "name": "✗ 機械翻訳エラー（直訳的）",
            "source": "彼の妻は彼のためにドアを開けた。",
            "prediction": "His wife door opened for him thing did.",
            "reference": "His wife opened the door for him."
        },
        {
            "name": "✗ 逆翻訳（英語→日本語として翻訳）",
            "source": "His wife opened the door for him.",
            "prediction": "彼の妻は彼のためにドアを開けた。",
            "reference": "His wife opened the door for him."
        }
    ]

    print("Testing predefined examples:\n")

    for i, example in enumerate(examples, 1):
        print(f"\n{'=' * 80}")
        print(f"Example {i}: {example['name']}")
        print(f"{'=' * 80}")
        print(f"Source:     {example['source']}")
        print(f"Reference:  {example['reference']}")
        print(f"Prediction: '{example['prediction']}'")
        print()

        results = evaluate_translation(
            example['source'],
            example['prediction'],
            example['reference']
        )

        print(format_score_display(results))
        print()  # 追加の空行で見やすく

    print("\n" + "=" * 80)
    print("Evaluation Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()