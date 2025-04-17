import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

repo_path = Path(__file__).resolve().parent.parent

def analyze_labels(label_dir, class_names, img_width=640, img_height=480):
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
                w = float(parts[3]) * img_width   # Convert to pixels
                h = float(parts[4]) * img_height  # Convert to pixels

                class_counts[cls_id] += 1
                box_sizes[cls_id].append((w, h))

    stats = []
    all_sign_sizes = []  # Collect sizes for combined sign stats
    all_sign_count = 0

    for cls_id in sorted(class_counts.keys()):
        sizes = np.array(box_sizes[cls_id])
        count = class_counts[cls_id]
        mean_wh = sizes.mean(axis=0)
        median_wh = np.median(sizes, axis=0)
        std_wh = sizes.std(axis=0)
        min_wh = sizes.min(axis=0)
        max_wh = sizes.max(axis=0)

        stats.append({
            "class_id": cls_id,
            "class_name": class_names[cls_id],
            "count": count,
            "avg_width_px": round(mean_wh[0], 2),
            "avg_height_px": round(mean_wh[1], 2),
            "median_width_px": round(median_wh[0], 2),
            "median_height_px": round(median_wh[1], 2),
            "std_width_px": round(std_wh[0], 2),
            "std_height_px": round(std_wh[1], 2),
            "min_width_px": round(min_wh[0], 2),
            "min_height_px": round(min_wh[1], 2),
            "max_width_px": round(max_wh[0], 2),
            "max_height_px": round(max_wh[1], 2),
        })

        # Collect for combined sign stats (classes 0–8)
        if 0 <= cls_id <= 8:
            all_sign_sizes.extend(sizes)
            all_sign_count += count

    # Add combined stats for signs 0–8
    if all_sign_sizes:
        all_sign_sizes = np.array(all_sign_sizes)
        mean_wh = all_sign_sizes.mean(axis=0)
        median_wh = np.median(all_sign_sizes, axis=0)
        std_wh = all_sign_sizes.std(axis=0)
        min_wh = all_sign_sizes.min(axis=0)
        max_wh = all_sign_sizes.max(axis=0)

        combined_stats = {
            "class_id": -1,
            "class_name": "ALL_SIGNS",
            "count": all_sign_count,
            "avg_width_px": round(mean_wh[0], 2),
            "avg_height_px": round(mean_wh[1], 2),
            "median_width_px": round(median_wh[0], 2),
            "median_height_px": round(median_wh[1], 2),
            "std_width_px": round(std_wh[0], 2),
            "std_height_px": round(std_wh[1], 2),
            "min_width_px": round(min_wh[0], 2),
            "min_height_px": round(min_wh[1], 2),
            "max_width_px": round(max_wh[0], 2),
            "max_height_px": round(max_wh[1], 2),
        }

        stats.insert(0, combined_stats)  # Add to top of list

    if not stats:
        print("⚠️ No valid labels found in directory:", label_dir)
        return pd.DataFrame()  # Return empty DataFrame gracefully

    df = pd.DataFrame(stats)

    if "count" in df.columns:
        df = df.sort_values(by="class_id", ascending=False).reset_index(drop=True)

    return df

class_names = [
    "oneway",
    "highwayentrance",
    "stop",
    "roundabout",
    "parking",
    "crosswalk",
    "noentry",
    "highwayexit",  
    "prio",
    "trafficlight",
    "block",
    "girl",
    "car"
]

# -------------------- MAIN --------------------
name = 'TestSet2024'
root = repo_path / 'bfmc_data' / 'generated' / 'testsets' / name
train_label_dir = root / 'labels'

print("🔍 Analyzing YOLOv8 train labels...")
df_stats = analyze_labels(train_label_dir, class_names)

print("\n📊 Label Statistics (Train Set) in Pixels:")
print(df_stats.to_string(index=False))

output_path = repo_path / 'training' / 'runs' / 'stats' / name
os.makedirs(output_path, exist_ok=True)
df_stats.to_csv(output_path / "label_statistics.csv", index=False)
