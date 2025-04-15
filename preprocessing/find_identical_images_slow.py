import os
import shutil
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from pathlib import Path
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

root = repo_path / "bfmc_data" / "base" / "testsets"
a_name = "rf2024c"
b_name = "rf2024"
FOLDER_A = os.path.join(root, a_name, "images")
FOLDER_B = os.path.join(root, b_name, "images")
FOLDER_C = os.path.join(root, a_name + "_matched")

# Ensure output folder exists
os.makedirs(FOLDER_C, exist_ok=True)

# Load all images from Folder B into memory (as arrays)
print("Loading images from Folder B into memory...")
images_b = []

for filename in os.listdir(FOLDER_B):
    path = os.path.join(FOLDER_B, filename)
    img = cv2.imread(path)  # Loads in BGR format
    if img is not None:
        images_b.append((filename, img))
    else:
        print(f"⚠️ Skipped unreadable image: {path}")

print(f"Loaded {len(images_b)} images from Folder B.")

# Compare function for multiprocessing
def compare_image(filename_a):
    path_a = os.path.join(FOLDER_A, filename_a)
    img_a = cv2.imread(path_a)

    if img_a is None:
        return None

    for _, img_b in images_b:
        if img_a.shape != img_b.shape:
            continue
        if np.array_equal(img_a, img_b):
            shutil.move(path_a, os.path.join(FOLDER_C, filename_a))
            return filename_a
    return None

# Get list of files in Folder A
images_a = os.listdir(FOLDER_A)
print(f"Processing {len(images_a)} images in Folder A using {cpu_count()} cores...")

# Progress bar with imap_unordered
matches = []
with Pool(processes=cpu_count()) as pool:
    for result in tqdm(pool.imap_unordered(compare_image, images_a), total=len(images_a)):
        if result:
            matches.append(result)

# Final report
print(f"✅ Found and moved {len(matches)} matching images to {FOLDER_C}")
