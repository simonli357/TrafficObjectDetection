import os
import cv2
from pathlib import Path

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Input folders
root = repo_path / "bfmc_data" / "base" / "testsets"
name = "rf2024"
image_dir = os.path.join(root, name, "images")
label_dir = os.path.join(root, name, "labels")

flipped_image_dir = os.path.join(root, name+"_flipped", "images")
flipped_label_dir = os.path.join(root, name+"_flipped", "labels")

os.makedirs(flipped_image_dir, exist_ok=True)
os.makedirs(flipped_label_dir, exist_ok=True)

# List all images
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

skip_count = 0
skip_empty_label_count = 0
for img_file in image_files:
    img_path = os.path.join(image_dir, img_file)
    base_name, ext = os.path.splitext(img_file)
    label_file = base_name + ".txt"
    label_path = os.path.join(label_dir, label_file)

    if not os.path.exists(label_path):
        print(f"[SKIP] Label file not found for image: {img_file}")
        skip_count += 1
        continue

    with open(label_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
      print(f"[SKIP] Background image (empty label): {img_file}")
      skip_empty_label_count += 1
      continue
    skip = False
    new_lines = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            print(f"[WARNING] Malformed label in {label_file}: {line}")
            continue

        cls = int(parts[0])
        if cls == 4:
            print(f"[SKIP] Skipping {img_file} due to class 4 in labels")
            skip_count += 1
            skip = True
            break

        x, y, w, h = map(float, parts[1:])
        flipped_x = 1.0 - x
        new_line = f"{cls} {flipped_x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        new_lines.append(new_line)

    if skip or not new_lines:
        continue

    # Flip the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Failed to read image: {img_file}")
        continue

    flipped_img = cv2.flip(img, 1)

    # Save flipped image
    flipped_img_file = base_name + "_flipped" + ext
    flipped_img_path = os.path.join(flipped_image_dir, flipped_img_file)
    cv2.imwrite(flipped_img_path, flipped_img)

    # Save flipped labels
    flipped_label_file = base_name + "_flipped.txt"
    flipped_label_path = os.path.join(flipped_label_dir, flipped_label_file)
    with open(flipped_label_path, 'w') as f:
        f.write('\n'.join(new_lines) + '\n')

    print(f"[OK] Processed: {img_file} -> {flipped_img_file}")

print("done. number skipped: ", skip_count + skip_empty_label_count)