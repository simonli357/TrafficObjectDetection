import os
import yaml
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- CONFIGURATION ---
model_dir = os.path.join(repo_path, "training", "runs", "core0418b15", "weights")
yaml_path = os.path.join(repo_path, "config", "train_config.yaml")

device = 0  # or 'cuda:0'
# Load class names from YAML
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)
class_names = config['names']
num_classes = len(class_names)
# test: /home/slsecret/Downloads/bfmc_data/TestSetAll/
# get the last part of the path
testset_name = config['test'].split('/')[-1]
print(f"Testset name: {testset_name}")
os.makedirs(os.path.join(model_dir, "metrics_"+testset_name), exist_ok=True)

# Get all .pt files
pt_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
pt_files.sort()  # optional: sorts by epoch name

for pt_file in pt_files:
    pt_path = os.path.join(model_dir, pt_file)
    print(f"Validating {pt_file}...")

    model = YOLO(pt_path)

    # Run validation (no saving)
    metrics = model.val(
        device=device,
        data=yaml_path,
        split='test',
        save=False,
        save_txt=False,
        save_conf=False,
        project=None,   # disables project saving
        name=None,      # disables run directory creation
        exist_ok=True
    )

    # Collect metrics
    data = []
    for class_id in range(num_classes):
        row = {
            "Class": class_names[class_id],
            "Precision": metrics.box.p[class_id],
            "Recall": metrics.box.r[class_id],
            "mAP50": metrics.box.ap50[class_id],
            "mAP50-95": metrics.box.ap[class_id]
        }
        data.append(row)

    excluded_ids = {9, 10, 11, 12}
    included_ids = [i for i in range(num_classes) if i not in excluded_ids]
    overall_signs = {
        "Class": "signs_only",
        "Precision": sum(metrics.box.p[i] for i in included_ids) / len(included_ids),
        "Recall": sum(metrics.box.r[i] for i in included_ids) / len(included_ids),
        "mAP50": sum(metrics.box.ap50[i] for i in included_ids) / len(included_ids),
        "mAP50-95": sum(metrics.box.ap[i] for i in included_ids) / len(included_ids)
    }
    data.insert(0, overall_signs)
    
    excluded_class = 10  # roadblock
    overall = {
        "Class": "all",
        "Precision": (metrics.box.mp * num_classes - metrics.box.p[excluded_class]) / (num_classes - 1),
        "Recall": (metrics.box.mr * num_classes - metrics.box.r[excluded_class]) / (num_classes - 1),
        "mAP50": (metrics.box.map50 * num_classes - metrics.box.ap50[excluded_class]) / (num_classes - 1),
        "mAP50-95": (metrics.box.map * num_classes - metrics.box.ap[excluded_class]) / (num_classes - 1)
    }
    data.insert(0, overall)

    df = pd.DataFrame(data)
    csv_path = os.path.join(model_dir, "metrics_"+testset_name, f"{pt_file}_val_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}\n")
