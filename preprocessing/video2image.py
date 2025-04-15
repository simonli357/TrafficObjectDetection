import cv2
import os
import random
from tqdm import tqdm
from pathlib import Path

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_random_bg_crop(bg_folder, target_width, pad_height):
    bg_files = [f for f in os.listdir(bg_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not bg_files:
        raise ValueError(f"No background images found in folder: {bg_folder}")

    while True:
        bg_path = os.path.join(bg_folder, random.choice(bg_files))
        bg_img = cv2.imread(bg_path)
        if bg_img is None:
            continue

        bg_height, bg_width = bg_img.shape[:2]
        if bg_width < target_width or bg_height < pad_height:
            continue

        x = random.randint(0, bg_width - target_width)
        y = random.randint(0, bg_height - pad_height)
        crop = bg_img[y:y+pad_height, x:x+target_width]
        return crop

def video_to_images_with_bg_pad(video_path, output_folder, bg_folder=None, fps=4, target_width=640, target_height=480, pad=True, resize=True):
    os.makedirs(output_folder, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        return

    video_fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps <= 0:
        print("[ERROR] Invalid video FPS (0 or less). Cannot proceed.")
        return

    interval = int(video_fps / fps)
    if interval == 0:
        interval = 1
        print("[WARNING] Requested FPS is higher than video FPS. Defaulting interval to 1.")

    count = 0
    image_count = 0

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            success, frame = vidcap.read()
            if not success:
                break

            if count % interval == 0:
                final_frame = frame

                if resize:
                    original_height, original_width = frame.shape[:2]
                    scale = target_width / original_width
                    new_height = int(original_height * scale)
                    resized_frame = cv2.resize(frame, (target_width, new_height))

                    if pad and new_height < target_height:
                        pad_total = target_height - new_height
                        pad_top = pad_total // 2
                        pad_bottom = pad_total - pad_top

                        if not bg_folder:
                            raise ValueError("Background folder must be provided when pad=True")

                        bg_crop = get_random_bg_crop(bg_folder, target_width, pad_total)
                        bg_top = bg_crop[:pad_top, :, :]
                        bg_bottom = bg_crop[pad_top:, :, :]

                        final_frame = cv2.vconcat([bg_top, resized_frame, bg_bottom])
                    else:
                        final_frame = resized_frame
                        if final_frame.shape[0] > target_height:
                            final_frame = cv2.resize(final_frame, (target_width, target_height))

                image_filename = os.path.join(output_folder, f"{name}_frame_{image_count:05d}.jpg")
                cv2.imwrite(image_filename, final_frame)
                image_count += 1

            count += 1
            pbar.update(1)

    vidcap.release()
    print(f"[DONE] Saved {image_count} images to '{output_folder}'.")

# === Settings ===
path = repo_path / "bfmc_data" / "base" / "unprocessed" / "other" / "Records"
name = "vroom4"
file = name + ".avi"
output_name = "vroom"
bg_folder = repo_path / "bfmc_data" / "base" / "unprocessed" / "Xinya_backgroundIMG"

video_to_images_with_bg_pad(
    video_path=os.path.join(path, file),
    output_folder=os.path.join(path, output_name, "images"),
    bg_folder=bg_folder,
    fps=5,
    target_width=640,
    target_height=480,
    pad=True,
    resize=True
)
