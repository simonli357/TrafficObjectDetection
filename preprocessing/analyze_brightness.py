import cv2
import numpy as np
import random
import os
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_brightness_level(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness
  
def adjust_brightness(image, min_brightness=1, max_attempts=5):
    for _ in range(max_attempts):
        beta = -20
        adjusted = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
        brightness = get_brightness_level(adjusted)
        if brightness >= min_brightness:
            return adjusted
    # fallback if all attempts are too dark
    return cv2.convertScaleAbs(image, alpha=1.0, beta=abs(beta))

def get_contrast_level(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)
    return contrast
  
def adjust_contrast(image, min_contrast=7, max_attempts=5):
    for _ in range(max_attempts):
        alpha = random.uniform(0.75, 1.25)
        alpha = 0.4
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        contrast = get_contrast_level(adjusted)
        if contrast >= min_contrast:
            return adjusted

image_path = os.path.join(repo_path, "bfmc_data", "base", "crop", "highwayentrance", "frame_532_1_287_47.jpg")
image = cv2.imread(image_path)
brightness = get_brightness_level(image)
adjusted_image = adjust_brightness(image)
new_brightness = get_brightness_level(adjusted_image)
print(f"Image brightness level: {brightness:.2f}, Adjusted brightness level: {new_brightness:.2f}")
# contrast = get_contrast_level(image)
# adjusted_contrast_image = adjust_contrast(image)
# new_contrast = get_contrast_level(adjusted_contrast_image)
# print(f"Image contrast level: {contrast:.2f}, Adjusted contrast level: {new_contrast:.2f}")
# cv2.imshow("Original Image", image)
# cv2.imshow("Adjusted Image", adjusted_image)
# cv2.imshow("Adjusted Contrast Image", adjusted_contrast_image)
# cv2.waitKey(0)