from augmentations import *
import os
import math
import random
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
random.seed(357)

def identity(image, *args, **kwargs):
    return image
normal_augmentations = [
    [apply_pixelation, apply_defocus_blur],
    [increase_brightness, decrease_brightness, apply_sun, apply_rain],
    [increase_contrast, decrease_contrast, identity],
    [increase_saturation, decrease_saturation, identity],
    [apply_color_temperature, strong_color_shift, apply_motion_blur],
    [rotate, perspective_warp],
]
light_augmentations = [
    [apply_motion_blur],
    [apply_defocus_blur],
    [apply_pixelation],
    [increase_brightness, decrease_brightness],
    [increase_contrast, decrease_contrast],
    [increase_saturation, decrease_saturation],
]
girl_augmentations = [
    [apply_motion_blur],
    [apply_defocus_blur],
    [apply_pixelation],
    [increase_brightness, decrease_brightness],
    [increase_contrast, decrease_contrast],
    [increase_saturation, decrease_saturation],
]
car_augmentations = [
    [apply_motion_blur],
    [apply_defocus_blur],
    [apply_pixelation],
    [increase_brightness, decrease_brightness],
    [increase_contrast, decrease_contrast],
    [increase_saturation, decrease_saturation],
]

CLASS_NAMES = ["oneway", "highwayentrance", "stopsign", "roundabout", "park",
               "crosswalk", "noentry", "highwayexit", "prio", "light",
               "roadblock", "girl", "cars2"]

can_flips = [0.5, 0.5, 0.2, 0.5, 0.0, 0.5, 0.5, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5]

target_numbers = [12000, 12000, 12000, 12000, 12000, 12000, 12000, 12000, 12000, 18000, 2000, 12000, 4000]

def apply_random_from_group(image, group_id, augmentations, image_path=None):
    group = augmentations[group_id]
    func = random.choice(group)

    try:
        if func == strong_color_shift:
            return func(image, image_path)
        return func(image)
    except Exception as e:
        print(f"Error applying function from group {group_id}: {e}")
        return image

def process_single_image(args):
    filename, directory, output_dir, multiplier, can_flip, num_augments, augmentations = args
    path = os.path.join(directory, filename)
    image = cv2.imread(path)
    if image is None:
        return

    for i in range(multiplier):
        transformed = image.copy()
        if random.random() < can_flip:
            transformed = flip_lr(transformed)

        chosen_groups = random.sample(range(len(augmentations)), num_augments)
        for group in chosen_groups:
            if random.random() < can_flip:
                transformed = flip_lr(transformed)
            transformed = apply_random_from_group(transformed, group, augmentations, image_path=path)

        name, ext = os.path.splitext(filename)
        new_fname = f"{name}_aug{i+1}{ext}"
        cv2.imwrite(os.path.join(output_dir, new_fname), transformed)

def apply_transformations_to_directory(directory, output_dir, multiplier, can_flip, num_augments, augmentations):
    image_files = [f for f in os.listdir(directory)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    tasks = [(fname, directory, output_dir, multiplier, can_flip, num_augments, augmentations)
             for fname in image_files]

    with Pool(processes=cpu_count()-1) as pool:
        for _ in tqdm(pool.imap_unordered(process_single_image, tasks),
                      total=len(tasks), desc="Augmenting Images"):
            pass

if __name__ == "__main__":
    num_augments = 2

    for idx in range(len(CLASS_NAMES)):
        folder_path = os.path.join(repo_path, "bfmc_data", "base", "crop", CLASS_NAMES[idx])
        output_path = os.path.join(repo_path, "bfmc_data", "generated", "crop_augmented", CLASS_NAMES[idx])
        os.makedirs(output_path, exist_ok=True)

        originals = [f for f in os.listdir(folder_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        original_count = len(originals)
        multiplier = max(1, round(target_numbers[idx] / original_count))

        print(f"Found {original_count} images for '{CLASS_NAMES[idx]}'.")
        print(f"Applying {multiplier} augmentations per image to reach ~{target_numbers[idx]} outputs.")

        if CLASS_NAMES[idx] == "light":
            aug_list = light_augmentations
        elif CLASS_NAMES[idx] == "girl":
            aug_list = girl_augmentations
        elif CLASS_NAMES[idx] == "cars2":
            aug_list = car_augmentations
        else:
            aug_list = normal_augmentations

        apply_transformations_to_directory(folder_path,
                                        output_path,
                                        multiplier,
                                        can_flips[idx],
                                        num_augments,
                                        aug_list)
