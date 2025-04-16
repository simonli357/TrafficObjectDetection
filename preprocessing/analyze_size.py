import os
from PIL import Image
import numpy as np
from pathlib import Path

def analyze_image_widths(folder_path):
    widths = []

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        try:
            with Image.open(file_path) as img:
                widths.append(img.width)
        except Exception as e:
            print(f"Skipped {filename}: {e}")

    if not widths:
        print("No valid images found in the folder.")
        return

    widths = np.array(widths)
    print("Image Width Analysis:")
    print(f"Mean Width     : {np.mean(widths):.2f}")
    print(f"Max Width      : {np.max(widths)}")
    print(f"Min Width      : {np.min(widths)}")
    print(f"Std Dev Width  : {np.std(widths):.2f}")

repo_path = Path(__file__).resolve().parent.parent
folder_path = repo_path / "bfmc_data" / "generated" / "crop_augmented_resized" / "4"
analyze_image_widths(folder_path)
