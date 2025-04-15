#!/usr/bin/env python3
import os
import sys
import glob
import argparse
from PIL import Image
import numpy as np
from pathlib import Path

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_image_files(folder):
    # Add more extensions if needed.
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.tiff"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return files

def main():
    CLASS_NAMES = ["oneway", "highwayentrance", "stopsign", "roundabout", "park",
                "crosswalk", "noentry", "highwayexit", "prio", "light",
                "roadblock", "girl", "cars2"]

    id = 5
    input_folder = repo_path / "bfmc_data" / "generated" / "crop_augmented" / CLASS_NAMES[id]
    output_folder = repo_path / "bfmc_data" / "generated" / "crop_augmented_resized" / str(id)
    min_width = 20
    max_width = 200
    mu = None

    # Set desired mean (mu) if not provided. For a uniform distribution between min and max, the mean must be (min+max)/2.
    desired_mu = (min_width + max_width) / 2 if mu is None else mu
    if desired_mu != (min_width + max_width) / 2:
        print("Warning: For a uniform distribution between {} and {}, the mean must be {}. "
              "You provided mu = {}. Using mu = {}."
              .format(min_width, max_width, (min_width + max_width) / 2,
                      desired_mu, (min_width + max_width) / 2))
        desired_mu = (min_width + max_width) / 2

    os.makedirs(output_folder, exist_ok=True)

    # Retrieve list of image files in the input folder.
    image_files = get_image_files(input_folder)
    if not image_files:
        print("No images found in the folder:", input_folder)
        sys.exit(1)

    # Extract image dimensions (width, height) and gather info.
    images_info = []
    for filepath in image_files:
        try:
            with Image.open(filepath) as im:
                w, h = im.size
            images_info.append({"path": filepath, "width": w, "height": h})
        except Exception as e:
            print("Could not open {}. Skipping. Error: {}".format(filepath, e))

    if not images_info:
        print("No valid images found!")
        sys.exit(1)

    # Sort images by original width (smallest first)
    images_info.sort(key=lambda x: x["width"])
    n = len(images_info)
    print("Processing {} images...".format(n))

    # Compute the ideal uniform target widths (linearly spaced between min_width and max_width).
    # These are the desired widths if no image needed to be left unchanged.
    target_widths = np.linspace(min_width, max_width, n)

    # Process each image.
    for i, info in enumerate(images_info):
        orig_width = info["width"]
        orig_height = info["height"]

        # Compute the ideal target width for this image.
        computed_target = int(round(target_widths[i]))
        # To ensure we never upscale, take the minimum of computed target and original width.
        target_w = computed_target if computed_target <= orig_width else orig_width

        # Calculate the scaling factor and the corresponding new height to preserve aspect ratio.
        scale = target_w / orig_width
        target_h = int(round(orig_height * scale))

        try:
            with Image.open(info["path"]) as im:
                im_resized = im.resize((target_w, target_h), Image.LANCZOS)
                # Save resized image using the same filename in the output folder.
                base_name = os.path.basename(info["path"])
                output_path = os.path.join(output_folder, base_name)
                im_resized.save(output_path)
                print("Processed {}: original {}x{}, target {}x{}".format(
                    base_name, orig_width, orig_height, target_w, target_h))
        except Exception as e:
            print("Failed to process {}. Error: {}".format(info["path"], e))
    
    print("All images have been processed and saved in:", output_folder)

    # -------------------------------------------------------------------
    # 6. Compute and print stats for the resulting images in the output folder.
    # -------------------------------------------------------------------
    resized_widths = []
    for file in os.listdir(output_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                with Image.open(os.path.join(output_folder, file)) as img:
                    width, _ = img.size
                    resized_widths.append(width)
            except Exception as e:
                print(f"Error reading {file}: {e}")

    if resized_widths:
        print("\n=== Resized Image Stats ===")
        print(f"Total resized images: {len(resized_widths)}")
        print(f"Min width     : {min(resized_widths)} px")
        print(f"Max width     : {max(resized_widths)} px")
        print(f"Mean width    : {np.mean(resized_widths):.2f} px")
        print(f"Std deviation : {np.std(resized_widths):.2f} px")
        
        # Create a histogram with 10-pixel bin sizes
        hist, edges = np.histogram(resized_widths, bins=np.arange(min(resized_widths), max(resized_widths)+11, 10))
        print("\nWidth histogram (10 px bins):")
        for count, edge_start, edge_end in zip(hist, edges[:-1], edges[1:]):
            percentage = (count / sum(hist)) * 100
            print(f"{int(edge_start):4d}-{int(edge_end - 1):4d} px: {percentage:6.2f}%")
    else:
        print("No resized images found to analyze.")
if __name__ == "__main__":
    main()
