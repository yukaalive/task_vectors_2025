#!/usr/bin/env python3
"""
Simple script to test COMET scores for translation examples.
Usage: python test_comet_score.py
"""

from comet import download_model, load_from_checkpoint

def evaluate_comet(source, prediction, reference):
    """
    Evaluate COMET score for a single translation example.

    Args:
        source: Source language text (Japanese)
        prediction: Model's translation (English)
        reference: Ground truth translation (English)

    Returns:
        COMET score (float between 0 and 1)
    """
    # Load COMET model (cached after first run)
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)

    # Prepare data
    data = [{
        "src": source,
        "mt": prediction,  # machine translation
        "ref": reference
    }]

    # Get score
    result = model.predict(data, batch_size=1, gpus=1)
    score = result.scores[0]

    return score


def main():
    """Interactive mode to test COMET scores"""

    print("=" * 80)
    print("COMET Score Tester")
    print("=" * 80)
    print("\nEnter translation examples to see their COMET scores.")
    print("Press Ctrl+C to exit.\n")

    # Example translations to test
    examples = [
        {
            "name": "Perfect Match",
            "source": "彼の妻は彼のためにドアを開けた。",
            "prediction": "彼の妻は彼のためにドアを開けた。",
            "reference": "his wife opened the door for him ."
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

        score = evaluate_comet(
            example['source'],
            example['prediction'],
            example['reference']
        )

        # Color code the score
        if score >= 0.7:
            quality = "🟢 GOOD"
        elif score >= 0.4:
            quality = "🟠 MEDIUM"
        else:
            quality = "🔴 BAD"

        print(f"\n→ COMET Score: {score:.4f} {quality}\n")

    print("\n" + "=" * 80)
    print("Custom Testing Mode")
    print("=" * 80)

    while True:
        try:
            print("\n" + "-" * 80)
            source = input("\nEnter source (Japanese): ").strip()
            if not source:
                print("Empty input. Exiting.")
                break

            reference = input("Enter reference (English): ").strip()
            if not reference:
                print("Empty input. Exiting.")
                break

            prediction = input("Enter prediction (English): ").strip()
            if not prediction:
                print("Empty input. Exiting.")
                break

            score = evaluate_comet(source, prediction, reference)

            if score >= 0.7:
                quality = "🟢 GOOD"
            elif score >= 0.4:
                quality = "🟠 MEDIUM"
            else:
                quality = "🔴 BAD"

            print(f"\n→ COMET Score: {score:.4f} {quality}")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            break


if __name__ == "__main__":
    main()
