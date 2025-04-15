from ultralytics import YOLO
import os
import pandas as pd
import yaml
from pathlib import Path

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

MODEL = "core_nocity12"
model_path = os.path.join(repo_path, "models", MODEL+".pt")

yaml_path = os.path.join(repo_path, "config" , "train_config.yaml")

model = YOLO(model_path)

output_dir = os.path.join(repo_path, "training", "results", MODEL)
metrics = model.val(
    device=0,
    data=yaml_path,
    split='test',
    save=True,               # Save images with predictions
    save_txt=True,           # Save predictions to .txt files
    save_conf=True,          # Save confidences in txt
    project=output_dir,      # Ensures output goes to your "results" folder
    name='test_results',     # So it's saved under results/test_results
    exist_ok=True            # Overwrite if folder already exists
)

with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)
class_names = config['names']
num_classes = len(class_names)

# Gather data for each class
data = []
for class_id in range(num_classes):
    name = class_names[class_id]
    row = {
        "Class": name,
        "Precision": metrics.box.p[class_id],
        "Recall": metrics.box.r[class_id],
        "mAP50": metrics.box.maps[class_id],
        "mAP50-95": metrics.box.maps[class_id]
    }
    data.append(row)

# Add an "all" row (overall metrics)
overall = {
    "Class": "all",
    "Precision": metrics.box.p.mean(),
    "Recall": metrics.box.r.mean(),
    "mAP50": metrics.box.map50,
    "mAP50-95": metrics.box.map
}
data.insert(0, overall)

# Save to CSV
df = pd.DataFrame(data)
csv_output_path = os.path.join(output_dir, 'test_results', 'val_metrics.csv')
df.to_csv(csv_output_path, index=False)
print(f"Saved metrics to {csv_output_path}")