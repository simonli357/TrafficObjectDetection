import os
import shutil
import yaml
from pathlib import Path

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

testsets = [
    'rf2024',
    'rf2024a',
    'rf2024b',
    'rf2024c',
    'vroom',
    # 'car_test_padded',
    # 'frames_sim0407',
    # 'lab',
    # 'rf0309b',
    # 'team2021',
    # 'xinya',
]
output_name = 'TestSet2024'
def combine_testsets():
    output_base_dir = repo_path / 'bfmc_data' / 'generated'

    if not output_name or not output_base_dir or not testsets:
        print("[ERROR] Missing required keys in config: 'output_name', 'output_base_dir', or 'datasets'")
        return

    output_image_dir = os.path.join(output_base_dir, output_name, "images")
    output_label_dir = os.path.join(output_base_dir, output_name, "labels")

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    total_images = 0
    for dataset in testsets:
        img_dir = os.path.join(output_base_dir, dataset, "images")
        label_dir = os.path.join(output_base_dir, dataset, "labels")

        if not os.path.isdir(img_dir) or not os.path.isdir(label_dir):
            print(f"[WARNING] Skipping {dataset} - missing image or label directory.")
            continue

        for img_file in os.listdir(img_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            base_name, ext = os.path.splitext(img_file)
            label_file = base_name + ".txt"

            img_path = os.path.join(img_dir, img_file)
            label_path = os.path.join(label_dir, label_file)

            if not os.path.exists(label_path):
                print(f"[WARNING] Missing label for {img_file} in {dataset}")
                continue

            # Avoid collisions by prefixing with dataset name
            new_base = f"{dataset}_{base_name}"
            new_img_name = new_base + ext
            new_label_name = new_base + ".txt"

            shutil.copyfile(img_path, os.path.join(output_image_dir, new_img_name))
            shutil.copyfile(label_path, os.path.join(output_label_dir, new_label_name))

            total_images += 1

    print(f"\n✅ Done! Combined {total_images} images into:")
    print(f"   Images: {output_image_dir}")
    print(f"   Labels: {output_label_dir}")

if __name__ == "__main__":
    combine_testsets()
