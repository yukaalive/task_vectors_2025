import pickle

# Load one pickle file to check task names
with open('swallow_7B.pkl', 'rb') as f:
    data = pickle.load(f)

print("Task names in swallow_7B.pkl:")
print("=" * 60)
for task_name in data.keys():
    print(f"  - {task_name}")
