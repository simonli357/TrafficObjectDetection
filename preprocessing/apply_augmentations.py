from augmentations import *
import os
import math
import random
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Base repository path
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
random.seed(357)

# --- Augmentation functions (unchanged) ---

def apply_random_from_group(image, group_id, image_path=None):
    if group_id == 1:
        func = random.choice([strong_color_shift, adjust_brightness, apply_color_temperature, lambda x, *_: x])
        if func == strong_color_shift:
            return func(image, image_path)
        return func(image)
    elif group_id == 2:
        return apply_desaturation(image)
    elif group_id == 3:
        func = random.choice([adjust_contrast_blend, adjust_contrast, lambda x, *_:x])
        try:
            if func == apply_sun:
                return func(image)
            return func(image)
        except Exception as e:
            return image
    elif group_id == 4:
        try:
            return apply_motion_blur(image)
        except Exception:
            return image
    elif group_id == 5:
        try:
            return random.choice([apply_defocus_blur])(image)
        except Exception:
            return image
    elif group_id == 6:
        func = random.choice([apply_albumentations_enhancements, lambda x, *_: x, apply_rain, apply_sun])
        try:
            if func == strong_color_shift:
                return func(image, image_path)
            return func(image)
        except Exception:
            return image
    elif group_id == 7:
        return random.choice([rotate, perspective_warp])(image)
    elif group_id == 8:
        return random.choice([apply_pixelation])(image)
    return image

def process_single_image(args):
    filename, directory, output_dir, multiplier, can_flip, num_augments = args
    path = os.path.join(directory, filename)
    image = cv2.imread(path)
    if image is None:
        return

    for i in range(multiplier):
        transformed = image.copy()
        if random.random() < can_flip:
            transformed = flip_lr(transformed)

        chosen_groups = random.sample(range(1, 9), num_augments)
        for group in chosen_groups:
            if random.random() < can_flip:
                transformed = flip_lr(transformed)
            try:
                transformed = apply_random_from_group(transformed, group, image_path=path)
            except Exception as e:
                print(f"Error applying group {group} to {filename}: {e}")
                pass

        name, ext = os.path.splitext(filename)
        new_fname = f"{name}_aug{i+1}{ext}"
        cv2.imwrite(os.path.join(output_dir, new_fname), transformed)

def apply_transformations_to_directory(directory, output_dir, multiplier, can_flip, num_augments):
    image_files = [f for f in os.listdir(directory)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    tasks = [(fname, directory, output_dir, multiplier, can_flip, num_augments)
             for fname in image_files]

    with Pool(processes=cpu_count()-1) as pool:
        for _ in tqdm(pool.imap_unordered(process_single_image, tasks),
                      total=len(tasks), desc="Augmenting Images"):
            pass

CLASS_NAMES = ["oneway", "highwayentrance", "stopsign", "roundabout", "park",
               "crosswalk", "noentry", "highwayexit", "prio", "light",
               "roadblock", "girl", "cars2"]

can_flips = [0.5, 0.5, 0.2, 0.5, 0.0, 0.5, 0.5, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5]

target_numbers = [12000, 12000, 12000, 12000, 12000, 12000, 12000, 12000, 12000, 14000, 2000, 12000, 4000]

if __name__ == "__main__":
    num_augments = 2

    for idx in range(9):
        folder_path = os.path.join(repo_path, "bfmc_data", "base", "crop", CLASS_NAMES[idx])
        output_path = os.path.join(repo_path, "bfmc_data", "generated", "crop_augmented", CLASS_NAMES[idx])
        os.makedirs(output_path, exist_ok=True)

        originals = [f for f in os.listdir(folder_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        original_count = len(originals)
        multiplier = max(1, round(target_numbers[idx] / original_count))

        print(f"Found {original_count} images for '{CLASS_NAMES[idx]}'.")
        print(f"Applying {multiplier} augmentations per image to reach ~{target_numbers[idx]} outputs.")

        apply_transformations_to_directory(folder_path,
                                           output_path,
                                           multiplier,
                                           can_flips[idx],
                                           num_augments)
