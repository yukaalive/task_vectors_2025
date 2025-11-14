"""
Test script for Phase 1 UI
Verifies basic functionality without running full experiments
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_imports():
    """Test if all required imports work"""
    print("Testing imports...")

    try:
        import gradio as gr
        print("✅ Gradio imported successfully")
    except ImportError as e:
        print(f"❌ Gradio import failed: {e}")
        print("   Install with: pip install gradio")
        return False

    try:
        from core.experiments_config import MODELS_TO_EVALUATE, TASKS_TO_EVALUATE
        print(f"✅ Config imported: {len(MODELS_TO_EVALUATE)} models, {len(TASKS_TO_EVALUATE)} tasks")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False

    try:
        from scripts.experiments.main import run_main_experiment, get_new_experiment_id
        print("✅ Experiment functions imported")
    except ImportError as e:
        print(f"❌ Experiment import failed: {e}")
        return False

    try:
        from ui.phase1_main import ExperimentRunner, LogCapture, create_ui
        print("✅ UI components imported")
    except ImportError as e:
        print(f"❌ UI import failed: {e}")
        return False

    return True


def test_log_capture():
    """Test LogCapture functionality"""
    print("\nTesting LogCapture...")

    from ui.phase1_main import LogCapture

    log = LogCapture()

    # Test write
    log.write("Test line 1\n")
    log.write("Test line 2\n")

    content = log.get_all_content()
    if "Test line 1" in content and "Test line 2" in content:
        print("✅ LogCapture write/read works")
    else:
        print("❌ LogCapture write/read failed")
        return False

    # Test incremental read
    log.write("Test line 3\n")
    new_content = log.get_new_content()

    if "Test line 3" in new_content and "Test line 1" not in new_content:
        print("✅ LogCapture incremental read works")
    else:
        print("❌ LogCapture incremental read failed")
        return False

    return True


def test_experiment_id():
    """Test experiment ID generation"""
    print("\nTesting experiment ID generation...")

    try:
        from scripts.experiments.main import get_new_experiment_id
        exp_id = get_new_experiment_id()
        print(f"✅ Generated experiment ID: {exp_id}")

        # Verify it's a number
        int(exp_id)
        print("✅ Experiment ID is numeric")
        return True
    except Exception as e:
        print(f"❌ Experiment ID generation failed: {e}")
        return False


def test_ui_creation():
    """Test UI creation (without launching)"""
    print("\nTesting UI creation...")

    try:
        from ui.phase1_main import create_ui
        demo = create_ui()
        print("✅ UI created successfully")
        return True
    except Exception as e:
        print(f"❌ UI creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test configuration access"""
    print("\nTesting configuration...")

    from core.experiments_config import MODELS_TO_EVALUATE, TASKS_TO_EVALUATE

    print(f"Models available: {len(MODELS_TO_EVALUATE)}")
    for i, (model_type, variant) in enumerate(MODELS_TO_EVALUATE):
        print(f"  {i+1}. {model_type}-{variant}")

    print(f"\nTasks available: {len(TASKS_TO_EVALUATE)}")
    for i, task in enumerate(TASKS_TO_EVALUATE):
        print(f"  {i+1}. {task}")

    return True


def test_paths():
    """Test file paths"""
    print("\nTesting paths...")

    from scripts.utils import MAIN_RESULTS_DIR

    print(f"Main results directory: {MAIN_RESULTS_DIR}")

    if os.path.exists(MAIN_RESULTS_DIR):
        print("✅ Results directory exists")

        # List existing experiment IDs
        exp_dirs = [d for d in os.listdir(MAIN_RESULTS_DIR) if d.isdigit()]
        if exp_dirs:
            print(f"   Existing experiment IDs: {', '.join(sorted(exp_dirs))}")
        else:
            print("   No experiments found yet")
    else:
        print("⚠️  Results directory doesn't exist (will be created on first run)")

    return True


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("ICL Task Vectors UI - Phase 1 Tests")
    print("="*60)

    tests = [
        ("Imports", test_imports),
        ("LogCapture", test_log_capture),
        ("Experiment ID", test_experiment_id),
        ("Configuration", test_config),
        ("Paths", test_paths),
        ("UI Creation", test_ui_creation),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} raised exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! You can now run the UI:")
        print("   python launch_ui.py")
        return True
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before running the UI.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
