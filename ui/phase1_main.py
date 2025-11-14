"""
Phase 1: Complete UI for main.py's main() method
Supports all features of the original main() function:
- Single or multiple model execution
- Task selection
- Experiment ID management
- Skip already completed tasks
- Real-time logging
- Result persistence
"""

import gradio as gr
import sys
import os
import io
import threading
import time
from typing import List, Optional, Tuple
from contextlib import redirect_stdout, redirect_stderr

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.experiments_config import MODELS_TO_EVALUATE, TASKS_TO_EVALUATE
from scripts.experiments.main import run_main_experiment, get_new_experiment_id
from scripts.utils import MAIN_RESULTS_DIR

class LogCapture:
    """Capture stdout/stderr and provide real-time access"""
    def __init__(self):
        self.buffer = io.StringIO()
        self.position = 0

    def write(self, text):
        self.buffer.write(text)
        return len(text)

    def flush(self):
        pass

    def get_new_content(self) -> str:
        """Get only new content since last call"""
        current_position = self.buffer.tell()
        self.buffer.seek(self.position)
        new_content = self.buffer.read()
        self.position = current_position
        self.buffer.seek(current_position)
        return new_content

    def get_all_content(self) -> str:
        """Get all content"""
        return self.buffer.getvalue()


class ExperimentRunner:
    """Manages experiment execution with logging"""

    def __init__(self):
        self.is_running = False
        self.should_stop = False

    def run_experiments(
        self,
        selected_models: List[str],
        selected_tasks: List[str],
        experiment_id: str,
        auto_experiment_id: bool
    ):
        """
        Run experiments for selected models and tasks
        Yields log updates in real-time
        """
        self.is_running = True
        self.should_stop = False

        try:
            # Validate inputs
            if not selected_models:
                yield "❌ Error: Please select at least one model\n"
                return

            if not selected_tasks:
                yield "❌ Error: Please select at least one task\n"
                return

            # Determine experiment ID
            if auto_experiment_id:
                experiment_id = get_new_experiment_id()
                yield f"📋 Auto-generated Experiment ID: {experiment_id}\n"
            else:
                if not experiment_id.strip():
                    experiment_id = ""
                yield f"📋 Using Experiment ID: {experiment_id if experiment_id else '(default)'}\n"

            yield f"\n{'='*60}\n"
            yield f"🚀 Starting Experiment\n"
            yield f"{'='*60}\n"
            yield f"Models to evaluate: {len(selected_models)}\n"
            yield f"Tasks to evaluate: {len(selected_tasks)}\n"
            yield f"Results directory: outputs/results/main/{experiment_id or '(default)'}/\n"
            yield f"{'='*60}\n\n"

            # Parse selected models
            models_to_run = []
            for model_str in selected_models:
                model_type, model_variant = model_str.split("-", 1)
                models_to_run.append((model_type, model_variant))

            # Override TASKS_TO_EVALUATE temporarily
            import core.experiments_config as config
            original_tasks = config.TASKS_TO_EVALUATE
            config.TASKS_TO_EVALUATE = selected_tasks

            # Run experiments for each model
            total_models = len(models_to_run)
            for idx, (model_type, model_variant) in enumerate(models_to_run, 1):

                if self.should_stop:
                    yield f"\n⚠️ Experiment stopped by user at model {idx}/{total_models}\n"
                    break

                yield f"\n{'='*60}\n"
                yield f"📊 Model {idx}/{total_models}: {model_type}-{model_variant}\n"
                yield f"{'='*60}\n\n"

                # Setup log capture
                log_capture = LogCapture()

                # Run experiment in thread
                experiment_complete = threading.Event()
                experiment_error = None

                def run_in_thread():
                    nonlocal experiment_error
                    try:
                        with redirect_stdout(log_capture), redirect_stderr(log_capture):
                            run_main_experiment(
                                model_type=model_type,
                                model_variant=model_variant,
                                experiment_id=experiment_id
                            )
                    except Exception as e:
                        experiment_error = e
                    finally:
                        experiment_complete.set()

                # Start thread
                thread = threading.Thread(target=run_in_thread, daemon=True)
                thread.start()

                # Stream logs
                while not experiment_complete.is_set():
                    if self.should_stop:
                        yield f"\n⚠️ Stopping experiment...\n"
                        break

                    new_logs = log_capture.get_new_content()
                    if new_logs:
                        yield new_logs

                    time.sleep(0.5)

                # Get remaining logs
                final_logs = log_capture.get_new_content()
                if final_logs:
                    yield final_logs

                # Check for errors
                if experiment_error:
                    yield f"\n❌ Error in {model_type}-{model_variant}: {str(experiment_error)}\n"
                    import traceback
                    yield f"\nTraceback:\n{traceback.format_exc()}\n"
                elif not self.should_stop:
                    yield f"\n✅ Model {model_type}-{model_variant} completed!\n"

            # Restore original tasks
            config.TASKS_TO_EVALUATE = original_tasks

            # Final summary
            if not self.should_stop:
                yield f"\n{'='*60}\n"
                yield f"🎉 All experiments completed!\n"
                yield f"{'='*60}\n"
                yield f"Results saved to: outputs/results/main/{experiment_id or '(default)'}/\n"

        except Exception as e:
            yield f"\n❌ Fatal error: {str(e)}\n"
            import traceback
            yield f"\nTraceback:\n{traceback.format_exc()}\n"

        finally:
            self.is_running = False

    def stop(self):
        """Request to stop the experiment"""
        self.should_stop = True


# Global runner instance
runner = ExperimentRunner()


def create_ui():
    """Create Phase 1 Gradio UI"""

    # Prepare model choices
    model_choices = [f"{m[0]}-{m[1]}" for m in MODELS_TO_EVALUATE]

    with gr.Blocks(
        title="ICL Task Vectors - Phase 1",
        theme=gr.themes.Soft()
    ) as demo:

        gr.Markdown("# 🧠 ICL Task Vectors Experiment UI - Phase 1")
        gr.Markdown("""
        **Complete UI for `scripts/experiments/main.py`**

        This interface provides all functionality of the original `main()` method:
        - Run single or multiple models
        - Select tasks to evaluate
        - Automatic experiment ID generation
        - Skip already completed tasks
        - Real-time logging
        """)

        with gr.Row():
            # Left column: Configuration
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ Configuration")

                # Model selection
                model_checkboxes = gr.CheckboxGroup(
                    choices=model_choices,
                    label="Models to Evaluate",
                    value=[model_choices[0]],
                    info="Select one or more models (equivalent to running main() with different arguments)"
                )

                # Task selection
                task_checkboxes = gr.CheckboxGroup(
                    choices=TASKS_TO_EVALUATE,
                    label="Tasks to Evaluate",
                    value=TASKS_TO_EVALUATE,
                    info="Select tasks to run (default: all tasks in TASKS_TO_EVALUATE)"
                )

                # Experiment ID
                gr.Markdown("### 🔢 Experiment ID")
                auto_id_checkbox = gr.Checkbox(
                    label="Auto-generate Experiment ID",
                    value=True,
                    info="Automatically create new experiment ID (max + 1)"
                )

                experiment_id_input = gr.Textbox(
                    label="Custom Experiment ID",
                    placeholder="Leave empty for default",
                    interactive=True,
                    info="Used when auto-generate is OFF"
                )

                # Quick selection buttons
                gr.Markdown("### 🎯 Quick Actions")
                with gr.Row():
                    select_all_models_btn = gr.Button("Select All Models", size="sm")
                    select_all_tasks_btn = gr.Button("Select All Tasks", size="sm")

                with gr.Row():
                    clear_models_btn = gr.Button("Clear Models", size="sm")
                    clear_tasks_btn = gr.Button("Clear Tasks", size="sm")

                # Run/Stop buttons
                gr.Markdown("### ▶️ Execution")
                with gr.Row():
                    run_btn = gr.Button("🚀 Run Experiment", variant="primary", size="lg")
                    stop_btn = gr.Button("⏹️ Stop", variant="stop", size="lg")

            # Right column: Logs and Output
            with gr.Column(scale=2):
                gr.Markdown("## 📊 Experiment Output")

                # Status indicator
                status_box = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False,
                    max_lines=1
                )

                # Log output - Full logs only
                log_output = gr.Textbox(
                    label="📄 Full Logs",
                    lines=50,
                    max_lines=None,  # Unlimited - show all lines
                    autoscroll=True,
                    show_copy_button=True,
                    interactive=False,
                    placeholder="Logs will appear here when experiment starts..."
                )

        # Information section
        gr.Markdown("""
        ---
        ## 📖 How to Use

        ### Basic Usage
        1. **Select Models**: Choose one or more models to evaluate
        2. **Select Tasks**: Choose which tasks to run (default: all)
        3. **Configure Experiment ID**: Auto-generate or specify custom ID
        4. **Run**: Click "Run Experiment" button
        5. **Monitor**: Watch real-time logs in the output panel

        ### Features
        - ✅ **Multiple Models**: Run experiments on multiple models sequentially
        - ✅ **Task Selection**: Choose which tasks to evaluate
        - ✅ **Auto Skip**: Automatically skips already completed tasks
        - ✅ **Progress Tracking**: Real-time logs show current progress
        - ✅ **Result Persistence**: Results saved after each task completes
        - ✅ **Interruptible**: Stop experiment at any time

        ### Experiment ID
        - **Auto-generate ON**: Creates new ID as (max existing ID + 1)
        - **Auto-generate OFF**: Uses custom ID or default (no subdirectory)
        - Results are saved to: `outputs/results/main/<experiment_id>/`

        ### Tips
        - Start with a single model and few tasks for testing
        - Results are saved incrementally, so partial runs are preserved
        - Check `outputs/results/main/` for saved results
        - Each model creates a separate pickle file: `<model_type>_<model_variant>.pkl`
        """)

        # Event handlers
        def update_status_running():
            return "🔄 Running..."

        def update_status_ready():
            return "✅ Ready"

        def select_all_models():
            return model_choices

        def select_all_tasks():
            return TASKS_TO_EVALUATE

        def clear_models():
            return []

        def clear_tasks():
            return []

        def run_experiment(models, tasks, exp_id, auto_id):
            """Wrapper for running experiments with status updates"""
            if not models:
                yield "Ready", "❌ Error: Please select at least one model\n"
                return

            # Initialize accumulated logs
            accumulated_log = ""

            # Update status to Running
            yield "🔄 Running...", ""

            # Run experiments and yield logs with status
            for log_chunk in runner.run_experiments(models, tasks, exp_id, auto_id):
                # Accumulate logs
                accumulated_log += log_chunk

                # Yield: status, full accumulated log
                yield "🔄 Running...", accumulated_log

            # Update status to Completed
            yield "✅ Completed", accumulated_log

        # Connect events
        run_btn.click(
            fn=run_experiment,
            inputs=[model_checkboxes, task_checkboxes, experiment_id_input, auto_id_checkbox],
            outputs=[status_box, log_output]
        )

        stop_btn.click(
            fn=runner.stop,
            inputs=None,
            outputs=None
        )

        select_all_models_btn.click(
            fn=select_all_models,
            inputs=None,
            outputs=model_checkboxes
        )

        select_all_tasks_btn.click(
            fn=select_all_tasks,
            inputs=None,
            outputs=task_checkboxes
        )

        clear_models_btn.click(
            fn=clear_models,
            inputs=None,
            outputs=model_checkboxes
        )

        clear_tasks_btn.click(
            fn=clear_tasks,
            inputs=None,
            outputs=task_checkboxes
        )

    return demo


def main():
    """Launch the UI"""
    print("Starting ICL Task Vectors UI (Phase 1)...")
    print(f"Project root: {project_root}")
    print(f"Models available: {len(MODELS_TO_EVALUATE)}")
    print(f"Tasks available: {len(TASKS_TO_EVALUATE)}")

    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",  # Changed back to 0.0.0.0 for Docker compatibility
        server_port=7860,
        share=True,  # Enable Gradio share for public URL
        show_error=True,
        inbrowser=False
    )


if __name__ == "__main__":
    main()
