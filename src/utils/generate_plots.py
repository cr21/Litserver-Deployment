import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import glob

# Get the latest metrics file from logs/train/runs/*/csv_logs/version_*/metrics.csv
metrics_files = glob.glob('logs/train/runs/*/logs/csv_logs/version_*/metrics.csv')
print(metrics_files)
if not metrics_files:
    print("Error: No metrics files found.")
    sys.exit(1)
latest_metrics_file = max(metrics_files, key=os.path.getctime)  # Get the latest file

# Load the metrics data
try:
    data = pd.read_csv(latest_metrics_file)
    # print(data.head())
except FileNotFoundError:
    print(f"Error: The file {latest_metrics_file} was not found.")
    sys.exit(1)

# Check if the necessary columns exist
if not all(col in data.columns for col in ['epoch', 'train/acc', 'train/loss']):
    print("Error: Required columns are missing in the metrics file.")
    sys.exit(1)

# Generate test metrics table report in test_metrics.md
with open('test_metrics.md', 'w') as f:
    f.write("## Test Metrics\n")
    f.write("| Metric | Value |\n")
    f.write("|--------|-------|\n")
    f.write(f"| Test Accuracy | {data['test/acc'].iloc[-1]} |\n")  # Assuming last entry is the latest
    f.write(f"| Test Loss | {data['test/loss'].iloc[-1]} |\n")  # Assuming last entry is the latest

# Plot Training Accuracy
plt.figure(figsize=(10, 5))
plt.plot(data['epoch'], data['train/acc'], marker='o', color='b', label='Train Accuracy')
plt.title('Training Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(data['epoch'])
plt.grid()
plt.legend()
plt.savefig('train_acc_plot.png')
plt.close()

# Plot Training Loss
plt.figure(figsize=(10, 5))
plt.plot(data['epoch'], data['train/loss'], marker='o', color='r', label='Train Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(data['epoch'])
plt.grid()
plt.legend()
plt.savefig('train_loss_plot.png')
plt.close()

