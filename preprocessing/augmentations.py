import cv2
import numpy as np
import albumentations as A
import random
from albumentations.augmentations.transforms import RandomSunFlare, RandomFog, RandomRain, RandomSnow, RandomBrightnessContrast, HueSaturationValue
import os 

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

random.seed(357)
np.random.seed(357)

BASE_SIZE = 200  # base image size for which original parameters were tuned

def get_scale_factor(image, base_size=BASE_SIZE):
    h, w = image.shape[:2]
    current_diag = (h**2 + w**2) ** 0.5
    base_diag = (base_size**2 + base_size**2) ** 0.5
    return current_diag / base_diag

def apply_motion_blur(image):
    scale = get_scale_factor(image)
    # Scaled kernel size, clamped between 1 and 80
    min_k, max_k = int(1 * scale), int(40 * scale)
    kernel_size = random.randint(max(1, min_k), max(min_k + 1, max_k))
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0
    kernel /= kernel_size
    return cv2.filter2D(image, -1, kernel)

def apply_pixelation(image):
    scale = get_scale_factor(image)
    # Scaled downscaling range, smaller scale => less pixelation
    min_scale = max(0.02, 0.08 / scale)
    max_scale = min(1.0, 0.3 / scale)
    pixel_scale = random.uniform(min_scale, max_scale)
    # pixel_scale = 0.08 # most pixelated
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, int(w * pixel_scale)), max(1, int(h * pixel_scale))), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_defocus_blur(image):
    scale = get_scale_factor(image)
    # Compute scaled range safely, clamped to valid odd values
    min_k = int(7 * scale)
    max_k = int(31 * scale)
    # Ensure min_k and max_k are odd and >= 3
    min_k = max(3, min_k | 1)  # bitwise OR 1 makes odd
    max_k = max(min_k + 2, max_k | 1)  # ensure max > min, also odd
    # Create list of valid odd kernel sizes
    valid_kernels = [k for k in range(min_k, max_k + 1, 2) if k > 0 and k % 2 == 1]
    if not valid_kernels:
        return image  # fallback: skip blur if nothing valid
    ksize = random.choice(valid_kernels)
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def get_brightness_level(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def adjust_brightness(image, min_brightness=20, max_brightness=246, max_attempts=5):
    for _ in range(max_attempts):
        beta = random.randint(-20, 20)
        adjusted = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
        brightness = get_brightness_level(adjusted)
        if min_brightness <= brightness <= max_brightness:
            return adjusted
    # fallback: brighten but stay within limit
    adjusted = cv2.convertScaleAbs(image, alpha=1.0, beta=15)
    if get_brightness_level(adjusted) > max_brightness:
        return image  # fallback to original if it goes too bright
    return adjusted

def get_contrast_level(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.std(gray)

def adjust_contrast(image, min_contrast=15, max_contrast=70, max_attempts=5):
    for _ in range(max_attempts):
        alpha = random.uniform(0.75, 1.25)
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        contrast = get_contrast_level(adjusted)
        if min_contrast <= contrast <= max_contrast:
            return adjusted
    # fallback: slight increase without overdoing it
    adjusted = cv2.convertScaleAbs(image, alpha=1.1, beta=0)
    if get_contrast_level(adjusted) > max_contrast:
        return image
    return adjusted

def adjust_contrast_blend(image, min_contrast=20, max_contrast=70, max_attempts=5):
    for _ in range(max_attempts):
        contrast_factor = random.uniform(0.4, 0.8)  # more conservative range
        gray = np.full_like(image, np.mean(image, dtype=np.uint8))
        adjusted = cv2.addWeighted(image, contrast_factor, gray, 1 - contrast_factor, 0)
        contrast = get_contrast_level(adjusted)
        if min_contrast <= contrast <= max_contrast:
            return adjusted
    return image  # fallback to original if no valid result

def apply_desaturation(image):
    strength = random.uniform(0.3, 0.8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_three = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(image, 1 - strength, gray_three, strength, 0)

def apply_color_temperature(image):
    # Warm effect with random factor up to 0.4
    warm_factor = random.uniform(0.1, 0.4)
    img = image.astype(np.float32)
    img[..., 2] *= 1 + warm_factor
    img[..., 0] *= 1 - warm_factor
    return np.clip(img, 0, 255).astype(np.uint8)

def strong_color_shift(image, path):
    """Applies color-aware channel scaling and highlight boosting based on folder name."""

    # Map folder names to dominant color
    color_map = {
        'crosswalk': 'blue',
        'oneway': 'blue',
        'roundabout': 'blue',
        'park': 'blue',
        'stopsign': 'red',
        'noentry': 'red',
        'highwayentrance': 'green',
        'highwayexit': 'green',
        'prio': 'yellow',
        'roadblock': 'red'
    }
    
    path_lower = path.lower()

    # Attempt to find the known label from the path
    label = None
    for key in color_map:
        if key in path_lower:
            label = key
            break

    assert label is not None, f"Could not determine label from path: {path}"
    dominant = color_map[label]

    # Define channel multipliers
    if dominant == 'blue':
        r_scale = random.uniform(0.5, 0.7)
        g_scale = random.uniform(1.1, 1.3)
        b_scale = random.uniform(1.2, 1.7)
    elif dominant == 'red':
        r_scale = random.uniform(1.1, 1.3)
        g_scale = random.uniform(0.8, 1.0)
        b_scale = random.uniform(0.8, 1.0)
    elif dominant == 'yellow':
        r_scale = random.uniform(0.9, 1.1)
        g_scale = random.uniform(0.9, 1.1)
        b_scale = random.uniform(1.0, 1.3)
    elif dominant == 'green':
        r_scale = random.uniform(0.8, 1.0)
        g_scale = random.uniform(1.1, 1.3)
        b_scale = random.uniform(0.8, 1.0)
    else:
        r_scale = g_scale = b_scale = random.uniform(0.8, 1.2)

    # Apply channel scaling
    img = image.astype(np.float32)
    img[..., 2] *= r_scale  # Red
    img[..., 1] *= g_scale  # Green
    img[..., 0] *= b_scale  # Blue
    image = np.clip(img, 0, 255).astype(np.uint8)

    # Boost highlights
    threshold = random.randint(200, 240)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = hsv[..., 2] > threshold
    hsv[..., 2][mask] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
  
def apply_sun(image):
    h, w = image.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    center = (random.randint(0, w), random.randint(0, h // 2))
    radius = int(min(h, w) * random.uniform(0.2, 0.4))
    cv2.circle(overlay, center, radius, (255, 255, 255), -1)
    overlay = cv2.GaussianBlur(overlay, (151, 151), 0)
    alpha = random.uniform(0.3, 0.7)
    image = cv2.addWeighted(image, 1, overlay, alpha, 0)

    # Random ROI coordinates, now corrected
    x1, x2 = sorted([random.uniform(0.1, 0.4), random.uniform(0.6, 0.9)])
    y1, y2 = sorted([random.uniform(0.1, 0.4), random.uniform(0.3, 0.5)])

    # random radius based on image size
    radius = int(min(h, w) * random.uniform(0.1, 0.25))
    aug = A.Compose([
        RandomSunFlare(
            flare_roi=(x1, y1, x2, y2),
            angle_lower=random.uniform(0.3, 1.0),
            num_flare_circles_lower=6,
            num_flare_circles_upper=10,
            src_radius=radius,
            src_color=(255, 255, 255),
            always_apply=True
        )
    ])
    return aug(image=image)['image']

def apply_rain(image):
    # Simulates rain with randomized intensity
    aug = A.Compose([
        RandomRain(
            blur_value=random.choice([2, 3]),
            brightness_coefficient=random.uniform(0.75, 0.95),
            always_apply=True
        )
    ])
    return aug(image=image)['image']

def apply_albumentations_enhancements(image):
    # Random brightness, contrast, hue, saturation, and value adjustments
    aug = A.Compose([
        RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.3, p=1.0),
        HueSaturationValue(
            hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=1.0)
    ])
    return aug(image=image)['image']

def flip_lr(image):
    # Horizontal flip
    return cv2.flip(image, 1)

def rotate(image):
    # Rotate image randomly between -15 and +15 degrees
    h, w = image.shape[:2]
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def perspective_warp(image):
    h, w = image.shape[:2]
    max_offset = int(w * 0.5 / 2)

    offset = random.randint(0, max_offset)

    # Original corner points
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    # Perspective-shifted destination points
    pts2 = np.float32([
        [offset, 0],
        [w - offset, 0],
        [offset, h],
        [w - offset, h]
    ])

    # Perspective transform matrix
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(image, matrix, (w, h))

    # Get bounding rectangle from destination points
    x_min = int(min(pts2[:, 0]))
    x_max = int(max(pts2[:, 0]))
    y_min = int(min(pts2[:, 1]))
    y_max = int(max(pts2[:, 1]))

    # Crop the result
    cropped = warped[y_min:y_max, x_min:x_max]

    return cropped

# -----------------------------
# Main for Testing Augmentations
# -----------------------------

if __name__ == "__main__":
    image_path = os.path.join(repo_path, "bfmc_data", "base", "crop", "oneway", "63.jpg")
    image = cv2.imread(image_path)

    cv2.imshow('Original', image)
    cv2.imshow('Motion Blur', apply_motion_blur(image.copy()))
    cv2.imshow('Pixelated', apply_pixelation(image.copy()))
    cv2.imshow('Defocus Blur', apply_defocus_blur(image.copy()))
    cv2.imshow('Brightness Adjusted', adjust_brightness(image.copy()))
    cv2.imshow('Contrast Adjusted', adjust_contrast(image.copy()))
    cv2.imshow('Contrast Blend', adjust_contrast_blend(image.copy()))
    cv2.imshow('Desaturated', apply_desaturation(image.copy()))
    cv2.imshow('Color Temperature', apply_color_temperature(image.copy()))
    cv2.imshow('Rain', apply_rain(image.copy()))
    cv2.imshow('Sun', apply_sun(image.copy()))
    cv2.imshow('Color Enhancement', apply_albumentations_enhancements(image.copy()))
    cv2.imshow('Flip LR', flip_lr(image.copy()))
    cv2.imshow('Rotated', rotate(image.copy()))
    cv2.imshow('Perspective Warp', perspective_warp(image.copy()))
    # strong color shift
    cv2.imshow('Strong Color Shift', strong_color_shift(image.copy(), image_path))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
