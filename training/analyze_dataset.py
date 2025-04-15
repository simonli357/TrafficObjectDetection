import os
from pathlib import Path
from collections import defaultdict
import yaml
import numpy as np
import pandas as pd
import os 
from pathlib import Path

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

yaml_path = repo_path /  "config" / "train_config.yaml"

def load_config(yaml_file):
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def analyze_labels(label_dir, class_names):
    label_dir = Path(label_dir)
    label_files = list(label_dir.glob("*.txt"))

    class_counts = defaultdict(int)
    box_sizes = defaultdict(list)

    for file in label_files:
        with open(file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue  # skip malformed lines
                cls_id = int(float(parts[0]))
                w = float(parts[3])
                h = float(parts[4])

                class_counts[cls_id] += 1
                box_sizes[cls_id].append((w, h))

    stats = []

    for cls_id in sorted(class_counts.keys()):
        sizes = np.array(box_sizes[cls_id])
        count = class_counts[cls_id]
        mean_wh = sizes.mean(axis=0)
        min_wh = sizes.min(axis=0)
        max_wh = sizes.max(axis=0)
        stats.append({
            "class_id": cls_id,
            "class_name": class_names[cls_id],
            "count": count,
            "avg_width": round(mean_wh[0], 4),
            "avg_height": round(mean_wh[1], 4),
            "min_width": round(min_wh[0], 4),
            "min_height": round(min_wh[1], 4),
            "max_width": round(max_wh[0], 4),
            "max_height": round(max_wh[1], 4),
        })

    df = pd.DataFrame(stats)
    df = df.sort_values(by="count", ascending=False)
    return df

# -------------------- MAIN --------------------
config = load_config(yaml_path)
root = Path(config['path'])
train_label_dir = root / 'labels/train'
class_names = [v for k, v in sorted(config['names'].items())]

print("🔍 Analyzing YOLOv8 train labels...")
df_stats = analyze_labels(train_label_dir, class_names)

print("\n📊 Label Statistics (Train Set):")
print(df_stats.to_string(index=False))

# Optionally save to CSV
df_stats.to_csv("label_statistics.csv", index=False)
