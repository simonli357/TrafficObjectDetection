import os
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

datasets = [
    'datasets_city',
    'datasets_a',
    'datasets_bb',
    # 'datasets_c',
    # 'datasets_e',
    'datasets_g',
    'datasets_john',
    'datasets_s',
    'datasets_x',
    'frames0331',
    'frames0331b',
    'frames_sim0407',
]
output_name: 'datasets_allx3'

root_dir = repo_path / 'bfmc_data' / 'base' / 'datasets'
output_dir = repo_path / 'bfmc_data' / 'generated' / output_name

images_train = output_dir / 'images'
labels_train = output_dir / 'labels'

for p in [images_train, labels_train]:
    p.mkdir(parents=True, exist_ok=True)

# Helper to copy and avoid name collisions
def safe_copy(src_path, dst_dir, used_names, prefix=""):
    base = f"{prefix}_{src_path.stem}" if prefix else src_path.stem
    ext = src_path.suffix
    new_name = base
    i = 1
    while f"{new_name}{ext}" in used_names:
        new_name = f"{base}_{i}"
        i += 1
    used_names.add(f"{new_name}{ext}")
    dst_path = dst_dir / f"{new_name}{ext}"
    shutil.copy(src_path, dst_path)
    return new_name

used_image_names = set()

for folder_name in tqdm(datasets, desc="Combining datasets"):
    dataset_path = root_dir / folder_name
    image_dir = dataset_path / 'images'
    label_dir = dataset_path / 'labels'

    images = list(image_dir.glob('*.*'))
    for img_path in tqdm(images, desc=f"  Processing {folder_name}", leave=False):
        label_path = label_dir / (img_path.stem + '.txt')
        if label_path.exists():
            new_base = safe_copy(img_path, images_train, used_image_names, prefix=folder_name)
            label_dst_path = labels_train / f"{new_base}.txt"
            try:
                shutil.copy(label_path, label_dst_path)
            except Exception as e:
                print(f"❌ Failed to copy label for {img_path.name}: {e}")
                continue

print("✅ Dataset combination complete.")
