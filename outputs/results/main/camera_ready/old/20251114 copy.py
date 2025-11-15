import pickle
import numpy as np

# Load one pickle file to check the structure
with open('/home/yukaalive/2025workspace/task_vectors/21_icl_task_vectors/outputs/results/main/camera_ready/swallow_7B.pkl', 'rb') as f:
    data = pickle.load(f)

print("Keys in swallow_7B.pkl:")
for key in data.keys():
    print(f"  - {key}")

print("\n" + "="*50)

# Check translation tasks
for task_name in ['translation_ja_en', 'translation_en_ja']:
    if task_name in data:
        print(f"\n{task_name} metrics:")
        for metric_name, metric_value in data[task_name].items():
            if isinstance(metric_value, (int, float, np.integer, np.floating)):
                print(f"  {metric_name}: {metric_value}")
            elif isinstance(metric_value, np.ndarray):
                print(f"  {metric_name}: ndarray with shape {metric_value.shape}")
            else:
                print(f"  {metric_name}: {type(metric_value)}")
    else:
        print(f"\n{task_name}: NOT FOUND")

print("\n" + "="*50)
print("\nAll other tasks:")
for task_name in data.keys():
    if task_name not in ['translation_ja_en', 'translation_en_ja']:
        print(f"\n{task_name}:")
        for metric_name in data[task_name].keys():
            print(f"  - {metric_name}")