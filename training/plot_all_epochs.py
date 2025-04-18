import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

repo_path = Path(__file__).resolve().parent.parent

# --- CONFIGURATION ---
weights_dir = repo_path / "training" / "runs" / "core041620" / "weights"
yaml_path = repo_path / "config" / "train_config.yaml"
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)
class_names = config['names']
num_classes = len(class_names)
# test: /home/slsecret/Downloads/bfmc_data/TestSetAll/
# get the last part of the path
testset_name = config['test'].split('/')[-1]
print(f"Testset name: {testset_name}")
metrics_dir = os.path.join(weights_dir, "metrics_"+testset_name)
output_dir = os.path.join(weights_dir, "plots_"+testset_name)
os.makedirs(output_dir, exist_ok=True)

# --- Load all val_metrics CSVs ---
csv_files = [f for f in os.listdir(metrics_dir) if f.endswith("_val_metrics.csv")]
csv_files.sort()

metrics_dict = {}  # {epoch: dataframe}
for f in csv_files:
    match = re.search(r'epoch(\d+)', f)
    if match:
        epoch = int(match.group(1))
        df = pd.read_csv(os.path.join(metrics_dir, f))
        metrics_dict[epoch] = df

# --- Sort by epoch ---
metrics_dict = dict(sorted(metrics_dict.items()))

# --- Extract list of class names ---
first_df = next(iter(metrics_dict.values()))
class_names = first_df['Class'].tolist()

# --- Metric types to track ---
metric_types = ["Precision", "Recall", "mAP50", "mAP50-95"]

# --- Plot Overall (Class == 'all') ---
all_metrics = {m: [] for m in metric_types}
epochs = list(metrics_dict.keys())

for epoch, df in metrics_dict.items():
    row = df[df['Class'] == 'all'].iloc[0]
    for m in metric_types:
        all_metrics[m].append(row[m])

# Plot overall metrics
for m in metric_types:
    plt.figure()
    plt.plot(epochs, all_metrics[m], marker='o')
    plt.title(f"{m} over Epochs (All Classes)")
    plt.xlabel("Epoch")
    plt.ylabel(m)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"all_classes_{m.lower()}.png"))
    plt.close()

# --- Plot Per-Class Metrics ---
for class_name in class_names:
    if class_name == "all":
        continue  # skip the overall again

    per_class_metrics = {m: [] for m in metric_types}
    for epoch, df in metrics_dict.items():
        row = df[df['Class'] == class_name].iloc[0]
        for m in metric_types:
            per_class_metrics[m].append(row[m])

    for m in metric_types:
        plt.figure()
        plt.plot(epochs, per_class_metrics[m], marker='o')
        plt.title(f"{m} over Epochs ({class_name})")
        plt.xlabel("Epoch")
        plt.ylabel(m)
        plt.grid(True)
        plt.tight_layout()
        filename = f"{class_name.replace(' ', '_')}_{m.lower()}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

print(f"✅ Plots saved in: {output_dir}")
