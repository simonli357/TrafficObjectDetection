import os
import shutil
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Paths and dataset configuration
repo_path = Path(__file__).resolve().parent.parent

datasets = [
    'datasets_city', 
    'datasets_a', 
    'datasets_bb', 
    'datasets_c',
    'datasets_e', 
    'datasets_g', 
    'datasets_john', 
    'datasets_s',
]
output_name = 'datasets_0416c'

root_dir = repo_path / 'bfmc_data' / 'base' / 'datasets'
output_dir = repo_path / 'bfmc_data' / 'generated' / output_name

images_train = output_dir / 'images'
labels_train = output_dir / 'labels'

for p in [images_train, labels_train]:
    p.mkdir(parents=True, exist_ok=True)

# Thread-safe used name set
used_names = set()
used_names_lock = Lock()

def safe_copy_pair(img_path, label_path, dst_img_dir, dst_label_dir, prefix):
    base = f"{prefix}_{img_path.stem}"
    ext = img_path.suffix
    new_name = base
    i = 1

    with used_names_lock:
        while f"{new_name}{ext}" in used_names:
            new_name = f"{base}_{i}"
            i += 1
        used_names.add(f"{new_name}{ext}")

    try:
        shutil.copy(img_path, dst_img_dir / f"{new_name}{ext}")
        shutil.copy(label_path, dst_label_dir / f"{new_name}.txt")
        return True
    except Exception as e:
        print(f"❌ Error copying {img_path.name}: {e}")
        return False

def build_copy_tasks():
    tasks = []
    for folder in datasets:
        dataset_path = root_dir / folder
        img_dir = dataset_path / 'images'
        label_dir = dataset_path / 'labels'
        images = list(img_dir.glob('*.*'))
        for img_path in images:
            label_path = label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                tasks.append((img_path, label_path, images_train, labels_train, folder))
    return tasks

def parallel_copy():
    tasks = build_copy_tasks()
    print(f"📦 Copying {len(tasks)} image-label pairs across datasets...")

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(safe_copy_pair, img, lbl, img_dst, lbl_dst, prefix)
            for img, lbl, img_dst, lbl_dst, prefix in tasks
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Copying"):
            pass

    print("✅ Dataset combination complete.")

if __name__ == "__main__":
    parallel_copy()
