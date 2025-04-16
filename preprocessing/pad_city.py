import os
import random
from PIL import Image
from pathlib import Path
import multiprocessing
import tqdm

# Define repository and folder paths
repo_path = Path(__file__).resolve().parent.parent

source_folder = repo_path / 'bfmc_data' / 'base' / 'datasets' / 'datasets_city' / 'images'
background_folder = repo_path / 'bfmc_data' / 'base' / 'backgrounds'
labels_folder = repo_path / 'bfmc_data' / 'base' / 'datasets' / 'datasets_city' / 'labels'
output_folder = repo_path / 'bfmc_data' / 'base' / 'datasets' / 'datasets_city_padded' / 'images'
output_labels_folder = repo_path / 'bfmc_data' / 'base' / 'datasets' / 'datasets_city_padded' / 'labels'

os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_labels_folder, exist_ok=True)

# Constants for image dimensions
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
ORIGINAL_HEIGHT = 320
PAD_HEIGHT = TARGET_HEIGHT - ORIGINAL_HEIGHT  # 480 - 320 = 160

def get_random_patch(bg_img):
    """Resize and tile a background image if necessary and return a random patch of height PAD_HEIGHT."""
    if bg_img.height < PAD_HEIGHT:
        scale = PAD_HEIGHT / bg_img.height
        new_width = int(bg_img.width * scale)
        bg_img = bg_img.resize((new_width, PAD_HEIGHT))
    if bg_img.width < TARGET_WIDTH:
        num_tiles = (TARGET_WIDTH // bg_img.width) + 1
        tiled = Image.new('RGB', (bg_img.width * num_tiles, bg_img.height))
        for i in range(num_tiles):
            tiled.paste(bg_img, (i * bg_img.width, 0))
        bg_img = tiled
    max_x = bg_img.width - TARGET_WIDTH
    x = random.randint(0, max_x)
    patch = bg_img.crop((x, 0, x + TARGET_WIDTH, PAD_HEIGHT))
    return patch

def adjust_yolo_labels(label_path, output_path):
    """
    Adjust YOLO labels for the padded image.
    The new y center is computed by mapping the original center (scaled for ORIGINAL_HEIGHT)
    and shifting it upward by the pad (in pixel space). Similarly, the height is scaled.
    """
    if not os.path.exists(label_path):
        return
    adjusted_lines = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = parts
            x, y, w, h = map(float, [x, y, w, h])
            # Convert the original y coordinate (normalized for a 320px height) into the new space:
            # new_y = (PAD_HEIGHT + (y * ORIGINAL_HEIGHT)) / TARGET_HEIGHT
            new_y = (PAD_HEIGHT + (y * ORIGINAL_HEIGHT)) / TARGET_HEIGHT
            # Scale the bounding box height to the new image:
            new_h = h * (ORIGINAL_HEIGHT / TARGET_HEIGHT)
            adjusted_lines.append(f"{cls} {x:.6f} {new_y:.6f} {w:.6f} {new_h:.6f}")
    with open(output_path, 'w') as f:
        f.write('\n'.join(adjusted_lines))

def process_image(filename):
    """
    Process a single image: pad it with a random background patch,
    save the new image, and adjust its corresponding YOLO labels.
    """
    if not filename.lower().endswith(('png', 'jpg', 'jpeg')):
        return  # Skip non-image files

    try:
        # Load source image
        src_path = os.path.join(source_folder, filename)
        src_img = Image.open(src_path).convert('RGB')

        # Select a random background image
        bg_filenames = os.listdir(background_folder)
        if not bg_filenames:
            raise ValueError("Background folder is empty.")
        bg_filename = random.choice(bg_filenames)
        bg_path = os.path.join(background_folder, bg_filename)
        bg_img = Image.open(bg_path).convert('RGB')

        # Get a random patch from the background
        bg_patch = get_random_patch(bg_img)

        # Create new image with the pad on top and the original image at the bottom
        new_img = Image.new('RGB', (TARGET_WIDTH, TARGET_HEIGHT))
        new_img.paste(bg_patch, (0, 0))
        new_img.paste(src_img, (0, PAD_HEIGHT))

        # Save the new padded image
        out_img_path = os.path.join(output_folder, filename)
        new_img.save(out_img_path)

        # Adjust labels
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(labels_folder, label_filename)
        out_label_path = os.path.join(output_labels_folder, label_filename)
        adjust_yolo_labels(label_path, out_label_path)

    except Exception as e:
        print(f"Error processing {filename}: {e}")

if __name__ == '__main__':
    image_files = os.listdir(source_folder)
    total_files = len(image_files)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(process_image, image_files), total=total_files, desc="Processing images"):
            pass

    print("Done padding images and adjusting labels!")
