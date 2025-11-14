import pickle
import numpy as np

# Load one pickle file to check the structure
with open('/home/yukaalive/2025workspace/task_vectors/21_icl_task_vectors/outputs/results/main/camera_ready/swallow_7B.pkl', 'rb') as f:
    data = pickle.load(f)

print("=" * 70)
print("ALL TASK NAMES IN swallow_7B.pkl:")
print("=" * 70)
for i, key in enumerate(data.keys(), 1):
    print(f"{i:2}. {key}")

print("\n" + "=" * 70)
print("CHECKING FOR JESC AND EASY TASKS:")
print("=" * 70)

jesc_tasks = [k for k in data.keys() if 'jesc' in k.lower()]
easy_tasks = [k for k in data.keys() if 'easy' in k.lower()]

print(f"\nTasks with 'jesc': {len(jesc_tasks)}")
for task in jesc_tasks:
    print(f"  - {task}")

print(f"\nTasks with 'easy': {len(easy_tasks)}")
for task in easy_tasks:
    print(f"  - {task}")

print("\n" + "=" * 70)
print("TESTING format_task_name FUNCTION:")
print("=" * 70)

def format_task_name(task_name):
    """Format task name for display - add appropriate suffix for multi-token translation tasks"""
    if not task_name.startswith('translation_'):
        return task_name

    # If it ends with _single, keep as is
    if task_name.endswith('_single'):
        return task_name

    # Check if it already has jesc or easy suffix
    if '_jesc' in task_name:
        return task_name.replace('_jesc', '_jesc_multi')
    elif '_easy' in task_name:
        return task_name.replace('_easy', '_easy_multi')
    else:
        return task_name + '_multi'

print("\nApplying format_task_name to all tasks:")
for task_name in sorted(data.keys()):
    formatted = format_task_name(task_name)
    changed = " ✓ CHANGED" if task_name != formatted else ""
    print(f"{task_name:40} → {formatted}{changed}")