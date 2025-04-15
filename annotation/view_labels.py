import os
import random
import cv2
import numpy as np
import argparse
import re

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def save_image_grid(img_grid, start_index, end_index):
    filename = f"images_{start_index}_to_{end_index}.png"
    cv2.imwrite(filename, img_grid)

def display_images(label_files, start_index):
    displayed_images = []

    for i in range(start_index, start_index + 32):
        if i >= len(label_files):
            break

        label_file = label_files[i]

        with open(os.path.join(label_folder, label_file), "r") as file:
            labels = file.readlines()

        image_filename = os.path.splitext(label_file)[0] + ".jpg"
        image_path = os.path.join(image_folder, image_filename)

        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image {image_path}")
            continue

        img_height, img_width, _ = img.shape
        
        # Add the image filename to the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, image_filename, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        for label in labels:
            class_id, x_center, y_center, width, height = [float(x) for x in label.strip().split()]
            class_id = int(class_id)

            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)

            cv2.rectangle(img, (x1, y1), (x2, y2), class_colors[class_id], 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            text = class_names[class_id]
            text_size = cv2.getTextSize(text, font, 0.5, 2)[0]
            cv2.rectangle(img, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), class_colors[class_id], -1)
            cv2.putText(img, text, (x1, y1 - 4), font, 0.5, (255, 255, 255), 1)

        displayed_images.append(img)

    return displayed_images

def show_images(images, start_index):
    if len(images) == 0:
        return 0

    resized_images = [cv2.resize(img, (176, 176)) for img in images]

    grid_rows = 4
    grid_cols = 8
    img_grid = []

    for row in range(grid_rows):
        row_imgs = []
        for col in range(grid_cols):
            idx = row * grid_cols + col
            if idx < len(resized_images):
                row_imgs.append(resized_images[idx])
            else:
                row_imgs.append(np.zeros_like(resized_images[0]))
        img_grid.append(np.hstack(row_imgs))

    img_grid = np.vstack(img_grid)

    cv2.imshow("Image Grid", img_grid)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('d'):
            return 1
        elif key == ord('a'):
            return 2
        elif key == ord('s'):
            save_image_grid(img_grid, start_index, start_index + len(images) - 1)
            return 3
        elif key == ord('r'):
            return 4
        elif key == ord('q'):
            return 0

base_path = os.path.join(repo_path, "bfmc_data", "base", "datasets", "datasets_g")
image_folder = os.path.join(base_path, "images")
label_folder = os.path.join(base_path, "labels")

print("image_folder: ", image_folder)

# Get the list of label files
label_files = [f for f in os.listdir(label_folder) if f.endswith(".txt")]
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
label_files = sorted(label_files, key=natural_sort_key)

# Define the class names and corresponding colors
class_names = ["oneway", "highwayexit", "stop", "roundabout", "parking", "crosswalk", "noentry", "highwayentrance", "prio", "light", "block", "girl", "car"]
class_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(class_names))]

start_index = 0
print("number of files is ", len(label_files))
while True:
    images = display_images(label_files, start_index)
    action = show_images(images, start_index)

    if action == 1:  # 'd' is pressed
        start_index += 32
    elif action == 2:  # 'a' is pressed
        start_index = max(start_index - 32, 0)
    elif action == 3:  # 's' is pressed
        # The image grid has already been saved in the `show_images()` function.
        pass
    elif action == 0:  # 'q' is pressed or an empty image grid is shown
        cv2.destroyAllWindows()
        break
    else: #'r' is pressed, randomly increment index
        rd = np.random.randint(low=0, high=int(len(label_files)/32))
        print("'r' pressed, shift by ", rd)
        start_index += rd

