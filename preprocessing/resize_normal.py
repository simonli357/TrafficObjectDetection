#!/usr/bin/env python3
import os
import sys
import glob
import argparse
from PIL import Image
import numpy as np
from scipy.stats import truncnorm
from tqdm import tqdm
from pathlib import Path

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_image_files(folder):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.tiff"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return files

CLASS_NAMES = ["oneway", "highwayentrance", "stopsign", "roundabout", "park",
               "crosswalk", "noentry", "highwayexit", "prio", "light",
               "roadblock", "girl", "cars2"]

id = 5
input_folder = repo_path / "bfmc_data" / "generated" / "crop_augmented" / CLASS_NAMES[id]
output_folder = repo_path / "bfmc_data" / "generated" / "crop_augmented_resized" / str(id)

mean_val = 100
std_val = 60

lower_bound = 12
upper_bound = 270

a = (lower_bound - mean_val) / std_val
b = (upper_bound - mean_val) / std_val

os.makedirs(output_folder, exist_ok=True)

image_files = get_image_files(input_folder)
if not image_files:
    print("No images found in:", input_folder)
    sys.exit(1)

images_info = []
for filepath in image_files:
    try:
        with Image.open(filepath) as im:
            w, h = im.size
        images_info.append({"path": filepath, "width": w, "height": h})
    except Exception as e:
        print(f"Could not open {filepath}: {e}")

if not images_info:
    print("No valid images found!")
    sys.exit(1)

images_info.sort(key=lambda x: x["width"])
n = len(images_info)
print(f"Processing {n} images...")

for i, info in enumerate(tqdm(images_info, desc="Resizing images", unit="img")):
    orig_width = info["width"]
    orig_height = info["height"]

    p = (i + 0.5) / n

    target_width = truncnorm.ppf(p, a, b, loc=mean_val, scale=std_val)
    
    target_w = int(round(target_width))
    if target_w > orig_width:
        target_w = orig_width

    scale = target_w / orig_width
    target_h = int(round(orig_height * scale))
    
    try:
        with Image.open(info["path"]) as im:
            im_resized = im.resize((target_w, target_h), Image.LANCZOS)
            base_name = os.path.basename(info["path"])
            output_path = os.path.join(output_folder, base_name)
            im_resized.save(output_path)
            # print(f"Processed {base_name}: original {orig_width}x{orig_height}, target {target_w}x{target_h}")
    except Exception as e:
        print(f"Failed to process {info['path']}: {e}")

print("All images have been processed and saved in:", output_folder)

resized_widths = []
for file in os.listdir(output_folder):
    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        try:
            with Image.open(os.path.join(output_folder, file)) as img:
                width, _ = img.size
                resized_widths.append(width)
        except Exception as e:
            print(f"Error reading {file}: {e}")

if resized_widths:
    print("\n=== Resized Image Stats ===")
    print(f"Total resized images: {len(resized_widths)}")
    print(f"Min width     : {min(resized_widths)} px")
    print(f"Max width     : {max(resized_widths)} px")
    print(f"Mean width    : {np.mean(resized_widths):.2f} px")
    print(f"Std deviation : {np.std(resized_widths):.2f} px")
    
    hist, edges = np.histogram(resized_widths, bins=np.arange(min(resized_widths), max(resized_widths)+11, 10))
    print("\nWidth histogram (10 px bins):")
    for count, edge_start, edge_end in zip(hist, edges[:-1], edges[1:]):
        percentage = (count / sum(hist)) * 100
        print(f"{int(edge_start):4d}-{int(edge_end - 1):4d} px: {percentage:6.2f}%")
else:
    print("No resized images found to analyze.")
