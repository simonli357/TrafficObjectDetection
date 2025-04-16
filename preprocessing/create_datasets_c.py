import os
import cv2
import numpy as np
import pandas as pd
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

repo_path = Path(__file__).resolve().parent.parent

random.seed(357)
np.random.seed(357)

# --- CONFIGURATION ---
DATASET_DIR = os.path.join(repo_path, "bfmc_data", "base", "datasets", "datasets_c")
CROPPED_DIR = os.path.join(repo_path, "bfmc_data", "generated", "crop_augmented_resized")
NUM_INSTANCES_PER_IMAGE = 3
NUM_THREADS = 7
TOTAL_SYNTHETIC_IMAGES = 30955

# --- Count class images automatically ---
classFolders = sorted(
    [d for d in os.listdir(CROPPED_DIR) if d.isdigit()],
    key=lambda x: int(x)
)
numClasses = len(classFolders)
classCounts = np.array([
    len([f for f in os.listdir(os.path.join(CROPPED_DIR, c)) if f.endswith(('.jpg', '.png'))])
    for c in classFolders
])
classIndices = np.cumsum(classCounts)

print("Detected classCounts:", classCounts)

# --- Track per-class usage ---
IdxCount = np.zeros(numClasses)
unique_numbers = np.load(os.path.join(repo_path, "bfmc_data", "generated", "unique_numbers.npy")).tolist()
unique_numbers_lock = threading.Lock()

def compute_intersection_area(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[2], boxB[2])
    xB = min(boxA[1], boxB[1])
    yB = min(boxA[3], boxB[3])
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    return interWidth * interHeight

def overlap(boxA, boxB, threshold=0.32):
    inter_area = compute_intersection_area(boxA, boxB)
    areaA = (boxA[1] - boxA[0]) * (boxA[3] - boxA[2])
    areaB = (boxB[1] - boxB[0]) * (boxB[3] - boxB[2])

    if areaA == 0 or areaB == 0:
        return False

    return (inter_area / areaA > threshold) or (inter_area / areaB > threshold)

def insert_objects(background, base_label_path, cropped_imgs, img_classes, image_index, show=False):
    label_path = os.path.join(DATASET_DIR, "labels", f"{image_index}.txt")
    image_path = os.path.join(DATASET_DIR, "images", f"{image_index}.jpg")
    
    existing_boxes = []
    if os.path.exists(label_path):
        df = pd.read_csv(label_path, sep=' ', header=None)
        for row in df.itertuples(index=False):
            xc, yc, w, h = row[1:5]
            left = int((xc - w / 2) * background.shape[1])
            right = int((xc + w / 2) * background.shape[1])
            top = int((yc - h / 2) * background.shape[0])
            bottom = int((yc + h / 2) * background.shape[0])
            existing_boxes.append([left, right, top, bottom])

    with open(label_path, 'a') as label_file:
        for img, cls in zip(cropped_imgs, img_classes):
            h_bg, w_bg = background.shape[:2]
            h_img, w_img = img.shape[:2]

            if w_img > w_bg or h_img > h_bg:
                scale = min(w_bg / w_img /3 , h_bg / h_img/3)
                original_shape = img.shape
                img = cv2.resize(img, (int(w_img * scale), int(h_img * scale)))
                print(f"Cropped image larger than background {image_index}: {original_shape} -> {img.shape}")
                h_img, w_img = img.shape[:2]
            stuck = False
            start_time = time.time()

            while True:
                if time.time() - start_time > 10:
                    stuck = True
                    print("stuck, skipping...")
                    break

                x_offset = random.randint(0, w_bg - w_img)
                y_offset = random.randint(0, h_bg - h_img)
                left, right = x_offset, x_offset + w_img
                top, bottom = y_offset, y_offset + h_img

                candidate_box = [left, right, top, bottom]
                if any(overlap(candidate_box, b) for b in existing_boxes):
                    continue

                # Accept placement
                existing_boxes.append(candidate_box)
                background[top:bottom, left:right] = img

                # Create YOLO label
                cx = (left + right) / 2 / w_bg
                cy = (top + bottom) / 2 / h_bg
                w = w_img / w_bg
                h = h_img / h_bg
                label_file.write(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

                if show:
                    cv2.rectangle(background, (left, top), (right, bottom), (0, 255, 0), 2)
                break

    if show:
        cv2.imshow('Preview', background)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.imwrite(image_path, background)

def get_image_class_pairs(sample_indices):
    img_classes, img_indices = [], []

    for global_idx in sample_indices:
        for cls_id in range(numClasses):
            if global_idx < classIndices[cls_id]:
                img_classes.append(cls_id)
                local_index = random.randint(1, classCounts[cls_id])
                img_indices.append(local_index)
                break

    return np.array(img_classes), np.array(img_indices)

def load_cropped_images(img_classes, img_indices):
    imgs = []
    for cls_id, idx in zip(img_classes, img_indices):
        path = os.path.join(CROPPED_DIR, str(cls_id), f"{idx}.jpg")
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Missing image: {path}")
        imgs.append(img)
    return imgs

def generate_synthetic_image(index):
    global unique_numbers, IdxCount
    with unique_numbers_lock:
        if len(unique_numbers) < NUM_INSTANCES_PER_IMAGE:
            unique_numbers = random.sample(range(int(classIndices[-1])), int(classIndices[-1]))
        sample_indices = [unique_numbers.pop() for _ in range(NUM_INSTANCES_PER_IMAGE)]

    img_classes, img_indices = get_image_class_pairs(sample_indices)
    for cls_id in img_classes:
        IdxCount[int(cls_id)] += 1

    bg_path = repo_path / "bfmc_data" / "base" / "datasets_bg" / "datasets_c" / "images" / f"{index}.jpg"
    background = cv2.imread(bg_path)
    if background is None:
        raise FileNotFoundError(f"Missing background: {bg_path}")

    cropped_imgs = load_cropped_images(img_classes, img_indices)
    insert_objects(background, DATASET_DIR, cropped_imgs, img_classes, index)
    return index


if __name__ == "__main__":
    print("Starting generation with", TOTAL_SYNTHETIC_IMAGES, "images...")

    # Create destination folders
    train_img_dir = os.path.join(DATASET_DIR, "images")
    train_lbl_dir = os.path.join(DATASET_DIR, "labels")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)

    src_lbl_dir = repo_path / "bfmc_data" / "base" / "datasets_bg" / "datasets_c" / "labels"
    if os.path.exists(src_lbl_dir):
        for fname in os.listdir(src_lbl_dir):
            if fname.endswith(".txt"):
                src_path = os.path.join(src_lbl_dir, fname)
                dst_path = os.path.join(train_lbl_dir, fname)
                with open(src_path, 'r') as src_file, open(dst_path, 'w') as dst_file:
                    dst_file.write(src_file.read())
        print(f"Copied existing label files from {src_lbl_dir} to {train_lbl_dir}.")
    else:
        print(f"⚠️  Warning: {src_lbl_dir} does not exist. No label files copied.")

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(generate_synthetic_image, i) for i in range(TOTAL_SYNTHETIC_IMAGES)]
        for future in as_completed(futures):
            i = future.result()
            if i % 100 == 0:
                print(f"Completed {i} images...")

    print("✅ Done. Class usage summary:")
    for i, count in enumerate(IdxCount):
        print(f"Class {i}: {int(count)}")