#!/usr/bin/env python3
"""
Simple script to test COMET and chrF scores for translation examples.
Usage: python test_comet_chrf_score.py
"""
import warnings
warnings.filterwarnings('ignore')
from comet import download_model, load_from_checkpoint
import sacrebleu
from langdetect import detect, LangDetectException


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
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    
    data = [{
        "src": source,
        "mt": prediction,
        "ref": reference
    }]
    
    comet_result = model.predict(data, batch_size=1, gpus=1)
    results['comet'] = comet_result.scores[0]
    
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
    print()

    examples = [
        {
            "name": "正解の例",
            "source": "彼の妻は彼のためにドアを開けた。",
            "prediction": "His wife opened the door for him.",
            "reference": "His wife opened the door for him."
        },
        {
            "name": "ほとんど正解の例",
            "source": "彼の妻は彼のためにドアを開けた。",
            "prediction": "His spouse opened the door for him.",
            "reference": "His wife opened the door for him."
        },
        {
            "name": "間違いの例",
            "source": "彼の妻は彼のためにドアを開けた。",
            "prediction": "His wif opened the door for cats.",
            "reference": "His wife opened the door for her."
        },
        {
            "name": "翻訳できていない例",
            "source": "彼の妻は彼のためにドアを開けた。",
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
        print(f"Prediction: {example['prediction']}")
        print()

        results = evaluate_translation(
            example['source'],
            example['prediction'],
            example['reference']
        )
        
        print(format_score_display(results))

    print("\n" + "=" * 80)
    print("Evaluation Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()