#!/usr/bin/env python3
import os
import sys
import glob
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from scipy.stats import truncnorm
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_image_files(folder):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.tiff"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return files

def compute_trunc_params(mean_val, std_val, min_w, max_w):
    a = (min_w - mean_val) / std_val
    b = (max_w - mean_val) / std_val
    return a, b

def resize_and_save(task):
    """
    task is a tuple:
    (info_dict, index, total_n, class_id, mean_val, std_val, a, b, output_folder)
    """
    info, i, n, cls, mean_val, std_val, a, b, output_folder = task
    orig_w, orig_h = info["width"], info["height"]
    p = (i + 0.5) / n

    # sample target dimension
    if cls in (11, 12):
        target_f = truncnorm.ppf(p, a, b, loc=mean_val, scale=std_val)
        target_h = min(int(round(target_f)), orig_h)
        scale = target_h / orig_h
        target_w = int(round(orig_w * scale))
    else:
        target_f = truncnorm.ppf(p, a, b, loc=mean_val, scale=std_val)
        target_w = min(int(round(target_f)), orig_w)
        scale = target_w / orig_w
        target_h = int(round(orig_h * scale))

    try:
        with Image.open(info["path"]) as im:
            im_resized = im.resize((target_w, target_h), Image.LANCZOS)
            new_name = f"{i+1}.jpg"
            out_path = os.path.join(output_folder, new_name)
            im_resized.save(out_path)
        return target_w
    except Exception as e:
        print(f"[ERROR] {info['path']}: {e}")
        return None

def process_class(id, repo_path):
    CLASS_NAMES = ["oneway","highwayentrance","stopsign","roundabout","park",
                   "crosswalk","noentry","highwayexit","prio","light",
                   "roadblock","girl","cars2"]
    CLASS_MEANS = [40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
               300, 150, 35]
    CLASS_STDS = [40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                40, 40, 40]
    CLASS_MIN_WIDTHS = [12, 12, 12, 12, 12, 12, 12, 12, 12, 16,
                        36, 30, 24]
    CLASS_MAX_WIDTHS = [120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
                        100, 200, 80]

    name      = CLASS_NAMES[id]
    mean_val  = CLASS_MEANS[id]
    std_val   = CLASS_STDS[id]
    min_w     = CLASS_MIN_WIDTHS[id]
    max_w     = CLASS_MAX_WIDTHS[id]

    inp = repo_path / "bfmc_data/generated/crop_augmented" / name
    out = repo_path / "bfmc_data/generated/crop_augmented_resized" / str(id)
    os.makedirs(out, exist_ok=True)

    files = get_image_files(inp)
    if not files:
        print(f"No images in {inp!r}")
        return

    images_info = []
    for fp in files:
        try:
            with Image.open(fp) as im:
                w,h = im.size
            images_info.append({"path": fp, "width": w, "height": h})
        except Exception as e:
            print(f"Skipping {fp}: {e}")

    if not images_info:
        print("No valid images, skipping.")
        return

    images_info.sort(key=lambda x: x["width"])
    n = len(images_info)
    print(f"[Class {id} – {name}] {n} images → resizing with {os.cpu_count()} workers")

    a, b = compute_trunc_params(mean_val, std_val, min_w, max_w)

    # build tasks
    tasks = [
        (info, i, n, id, mean_val, std_val, a, b, str(out))
        for i, info in enumerate(images_info)
    ]

    resized_widths = []
    # use ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor() as exe:
        # submit all
        futures = {exe.submit(resize_and_save, task): task for task in tasks}
        for fut in tqdm(as_completed(futures),
                        total=n, desc=f"Class {id} resizing", unit="img"):
            w = fut.result()
            if w is not None:
                resized_widths.append(w)

    # stats
    if resized_widths:
        print(f"\n[Class {id}] Stats for {len(resized_widths)} images:")
        print(f"  min: {min(resized_widths)} px")
        print(f"  max: {max(resized_widths)} px")
        print(f"  mean: {np.mean(resized_widths):.2f} px")
        print(f"  std: {np.std(resized_widths):.2f} px")

        # histogram
        bins = np.arange(min(resized_widths), max(resized_widths)+11, 10)
        hist, edges = np.histogram(resized_widths, bins=bins)
        print("  width histogram (10 px bins):")
        total = sum(hist)
        for c, e0, e1 in zip(hist, edges[:-1], edges[1:]):
            print(f"    {int(e0):3d}-{int(e1-1):3d}: {c/total*100:5.2f}%")
    else:
        print("No resized images to report.")

def main():
    classes = list(range(13))

    repo_path = Path(__file__).resolve().parent.parent
    for cid in classes:
        process_class(cid, repo_path)

if __name__ == "__main__":
    main()
