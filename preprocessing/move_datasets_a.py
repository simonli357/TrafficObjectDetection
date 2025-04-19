import os
import shutil
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

repo_path = Path(__file__).resolve().parent.parent

name = "datasets_c"
src_images_dir = repo_path / "bfmc_data/generated/datasets_0416/images"
src_labels_dir = repo_path / "bfmc_data/generated/datasets_0416/labels"

dst_images_dir = repo_path / ("bfmc_data/base/datasets/"+name+"0416c/images")
dst_labels_dir = repo_path / ("bfmc_data/base/datasets/"+name+"0416c/labels")

os.makedirs(dst_images_dir, exist_ok=True)
os.makedirs(dst_labels_dir, exist_ok=True)

image_files = sorted([f for f in src_images_dir.glob(name+"_*.jpg")])
label_files = sorted([f for f in src_labels_dir.glob(name+"_*.txt")])
print(f"Found {len(image_files)} image files and {len(label_files)} label files.")
def copy_file_pair(image_path):
    filename = image_path.name  # e.g., datasets_a_000001.jpg
    base_name = filename.replace(name+"_", "")  # e.g., 000001.jpg

    # Copy image
    dst_img = dst_images_dir / base_name
    shutil.copy(image_path, dst_img)

    # Copy label if exists
    label_path = src_labels_dir / filename.replace(".jpg", ".txt")
    dst_label = dst_labels_dir / base_name.replace(".jpg", ".txt")
    if label_path.exists():
        shutil.copy(label_path, dst_label)

# Use multithreading and tqdm for progress
with ThreadPoolExecutor(max_workers=8) as executor:
    list(tqdm(executor.map(copy_file_pair, image_files), total=len(image_files), desc="Copying datasets_a_* files"))
