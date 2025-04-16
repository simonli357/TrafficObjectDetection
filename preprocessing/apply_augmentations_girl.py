from augmentations import *
import os
import math
import random
import cv2
from tqdm import tqdm

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

random.seed(357)

def apply_random_from_group(image, group_id, image_path=None):
    if group_id == 1:
        func = random.choice([adjust_brightness, apply_color_temperature, apply_motion_blur])
        return func(image)
    elif group_id == 2:
        return apply_desaturation(image)
    elif group_id == 3:
        return random.choice([adjust_contrast_blend, adjust_contrast])(image)
    elif group_id == 4:
        return apply_motion_blur(image)
    elif group_id == 5:
        return random.choice([apply_defocus_blur])(image)
    elif group_id == 6:
        func = random.choice([apply_albumentations_enhancements, lambda x, *_: x])
        return func(image)
    elif group_id == 7:
        return random.choice([rotate, perspective_warp])(image)
    elif group_id == 8:
        return random.choice([apply_pixelation])(image)
    return image


def apply_transformations_to_directory(directory, output_dir, multiplier, can_flip, num_augments):
    image_files = [
        f for f in os.listdir(directory)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    for filename in tqdm(image_files, desc="Augmenting Images"):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(directory, filename)
            image = cv2.imread(path)
            image1 = image.copy()
            if random.random() < can_flip:
                image1 = flip_lr(image)

            name, ext = os.path.splitext(filename)

            for i in range(multiplier):
                transformed = image.copy()
                chosen_groups = random.sample(range(1, 9), num_augments)
                for group in chosen_groups:
                    if random.random() < can_flip:
                        transformed = flip_lr(transformed)
                    transformed = apply_random_from_group(transformed, group, image_path=path)
                new_path = os.path.join(output_dir, f"{name}_aug{i+1}{ext}")
                cv2.imwrite(new_path, transformed)


CLASS_NAMES = ["oneway", "highwayentrance", "stopsign", "roundabout", "park",
               "crosswalk", "noentry", "highwayexit", "prio", "light",
               "roadblock", "girl", "cars2"]

can_flips = [0.5, 0.5, 0.2, 0.5, 0.0, 0.5, 0.5, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5]
target_numbers = [12000,12000,12000,12000,12000,12000,12000,12000,12000,12000,3000,12000,4000]

if __name__ == "__main__":
    id = 11
    num_augments = 2
    target_number = target_numbers[id]
    folder_path = os.path.join(repo_path, "bfmc_data", "base", "crop", CLASS_NAMES[id])
    output_path = os.path.join(repo_path, "bfmc_data", "generated", "crop_augmented", CLASS_NAMES[id])
    os.makedirs(output_path, exist_ok=True)

    original_image_count = len([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    multiplier = max(1, round(target_number / original_image_count))

    print(f"Found {original_image_count} images.")
    print(f"Applying {multiplier} augmentations per image to approximate {target_number} total.")

    apply_transformations_to_directory(folder_path, output_path, multiplier, can_flips[id], num_augments)
