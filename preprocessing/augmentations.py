import cv2
import numpy as np
import albumentations as A
import random
from albumentations.augmentations.transforms import RandomSunFlare, RandomFog, RandomRain, RandomSnow, RandomBrightnessContrast, HueSaturationValue
import os 
import math

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

random.seed(357)
np.random.seed(357)

BASE_SIZE = 200  # base image size for which original parameters were tuned

def get_scale_factor(image, base_size=BASE_SIZE):
    h, w = image.shape[:2]
    current_diag = (h**2 + w**2) ** 0.5
    base_diag = (base_size**2 + base_size**2) ** 0.5
    return current_diag / base_diag

def apply_motion_blur(image, testing=False):
    scale = get_scale_factor(image)
    # Scaled kernel size, clamped between 1 and 80
    min_k, max_k = int(15 * scale), int(57 * scale)
    kernel_size = random.randint(max(1, min_k), max(min_k + 1, max_k))
    if testing:
        kernel_size = max_k
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0
    kernel /= kernel_size
    return cv2.filter2D(image, -1, kernel)

def apply_pixelation(image, testing=False):
    scale = get_scale_factor(image)
    if testing:
        print(f"Scale factor: {scale:.3f}")
    # Scaled downscaling range, smaller scale => less pixelation
    min_scale = max(0.02, 0.08 / scale)
    max_scale = min(1.0, 0.15 / scale)
    pixel_scale = random.uniform(min_scale, max_scale)
    if testing:
        pixel_scale = min_scale    
        pixel_scale = max_scale 
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, int(w * pixel_scale)), max(1, int(h * pixel_scale))), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_defocus_blur(image, testing=False):
    scale = get_scale_factor(image)
    # Compute scaled range safely, clamped to valid odd values
    min_k = int(7 * scale)
    max_k = int(37 * scale)
    # Ensure min_k and max_k are odd and >= 3
    min_k = max(3, min_k | 1)  # bitwise OR 1 makes odd
    max_k = max(min_k + 2, max_k | 1)  # ensure max > min, also odd
    valid_kernels = [k for k in range(min_k, max_k + 1, 2) if k > 0 and k % 2 == 1]
    if not valid_kernels:
        return image  # fallback: skip blur if nothing valid
    ksize = random.choice(valid_kernels)
    if testing:
        ksize = max_k
        ksize = int((min_k+max_k)/2 * scale)
        if ksize % 2 == 0:
            ksize += 1
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

# helpers
def get_brightness_level(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)
def get_contrast_level(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.std(gray)

def increase_contrast(image, min_factor=1.15, max_factor=3.14, testing=False):
    max_brightness = 247
    max_aug_factor = 1 + 0.4
    max_allowed_brightness = math.floor(max_brightness / max_aug_factor)
    for _ in range(5):
        factor = random.uniform(min_factor, max_factor)
        if testing:
            factor = max_factor
        adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        brightness = get_brightness_level(adjusted)
        if brightness <= max_allowed_brightness:
            return adjusted
    current = get_brightness_level(image)
    if current < max_allowed_brightness:
        safe_factor = max_allowed_brightness / (current + 1e-5)
        safe_factor = min(safe_factor, max_factor)
        adjusted = cv2.convertScaleAbs(image, alpha=safe_factor, beta=0)
        if testing:
            print(f"Brightness increase fallback with factor={safe_factor:.3f}")
        return adjusted
    if testing:
        print("Brightness increase failed after max attempts.")
    return image

def decrease_contrast(image, min_factor=0.22, max_factor=0.753, testing=False):
    min_brightness = math.ceil(22 / (1-0.4))
    for _ in range(5):
        factor = random.uniform(min_factor, max_factor)
        if testing:
            factor = min_factor
            factor = 0.7
        adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        if get_brightness_level(adjusted) >= min_brightness:
            return adjusted
    current = get_brightness_level(image)
    if current > min_brightness:
        safe_factor = min_brightness / (current + 1e-5)  # avoid divide by zero
        safe_factor = max(safe_factor, min_factor)  # floor it
        adjusted = cv2.convertScaleAbs(image, alpha=safe_factor, beta=0)
        if testing:
            print(f"Brightness decrease fallback with factor={safe_factor:.3f}")
        return adjusted
    if testing:
        print("Brightness decrease failed after max attempts.")
    return image

def increase_brightness(image, min_beta=30, max_beta=150, max_attempts=5, testing=False):
    target_max= 247 / (1 + 0.4)
    current = get_brightness_level(image)
    for _ in range(max_attempts):
        beta = random.randint(min_beta, max_beta)
        if testing:
            beta = max_beta
        adjusted = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
        if get_brightness_level(adjusted) <= target_max:
            return adjusted
    safe_beta = int(target_max - current)
    safe_beta = min(safe_beta, max_beta)
    adjusted = cv2.convertScaleAbs(image, alpha=1.0, beta=safe_beta)
    if testing:
        print(f"[increase_brightness] Fallback beta={safe_beta}")
    return adjusted

def decrease_brightness(image, min_beta=15, max_beta=125, max_attempts=5, testing=False):
    target_min = 22 / (1 - 0.4)
    current = get_brightness_level(image)
    for _ in range(max_attempts):
        beta = random.randint(min_beta, max_beta)
        if testing:
            beta = max_beta
            # beta = min_beta
        adjusted = np.clip(image.astype(np.int16) - beta, 0, 255).astype(np.uint8)
        if get_brightness_level(adjusted) >= target_min:
            return adjusted
    safe_beta = int(current - target_min)
    # safe_beta = max(safe_beta, min_beta)
    adjusted = np.clip(image.astype(np.int16) - safe_beta, 0, 255).astype(np.uint8)
    if testing:
        print(f"[decrease_brightness] Fallback beta={-safe_beta}")
    return adjusted

def get_saturation_level(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[..., 1])

def increase_saturation(image, min_factor=1.5, max_factor=5.0, max_attempts=5, testing=False):
    target_max= 250 #/ (1 + 0.7)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    current = get_saturation_level(image)
    for _ in range(max_attempts):
        factor = random.uniform(min_factor, max_factor)
        if testing:
            factor = max_factor
            # factor = min_factor
        hsv_copy = hsv.copy()
        hsv_copy[..., 1] *= factor
        hsv_copy[..., 1] = np.clip(hsv_copy[..., 1], 0, 255)
        adjusted = cv2.cvtColor(hsv_copy.astype(np.uint8), cv2.COLOR_HSV2BGR)
        if testing:
            print(f"normal saturation: {get_saturation_level(image)}, adjusted saturation: {get_saturation_level(adjusted)}")
        if get_saturation_level(adjusted) <= target_max:
            return adjusted
    safe_factor = target_max / (current + 1e-5)
    safe_factor = min(safe_factor, max_factor)
    hsv[..., 1] *= safe_factor
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if testing:
        print(f"[increase_saturation] Fallback factor={safe_factor:.3f}")
    return adjusted

def decrease_saturation(image, min_factor=0.5, max_factor=0.95, max_attempts=5, testing=False):
    target_min= 7 / (1 - 0.7)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    current = get_saturation_level(image)
    for _ in range(max_attempts):
        factor = random.uniform(min_factor, max_factor)
        if testing:
            factor = min_factor
            factor = max_factor
        hsv_copy = hsv.copy()
        hsv_copy[..., 1] *= factor
        hsv_copy[..., 1] = np.clip(hsv_copy[..., 1], 0, 255)
        if testing:
            hsv_copy[..., 1] *= 0.3
            hsv_copy[..., 1] = np.clip(hsv_copy[..., 1], 0, 255)
        adjusted = cv2.cvtColor(hsv_copy.astype(np.uint8), cv2.COLOR_HSV2BGR)
        if testing:
            print(f"normal saturation: {get_saturation_level(image)}, adjusted saturation: {get_saturation_level(adjusted)}")
        if get_saturation_level(adjusted) >= target_min:
            return adjusted

    safe_factor = target_min / (current + 1e-5)
    safe_factor = max(safe_factor, min_factor)
    hsv[..., 1] *= safe_factor
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if testing:
        print(f"[decrease_saturation] Fallback factor={safe_factor:.3f}")
    return adjusted

def apply_color_temperature(image, testing=False):
    # Warm effect with random factor up to 0.4
    warm_factor = random.uniform(0.075, 0.42)
    if testing:
        # warm_factor = 0.5
        warm_factor = 0.12
    img = image.astype(np.float32)
    img[..., 2] *= 1 + warm_factor
    img[..., 0] *= 1 - warm_factor
    return np.clip(img, 0, 255).astype(np.uint8)

def strong_color_shift(image, path, testing=False):
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
        r_scale = random.uniform(0.57, 0.9)
        g_scale = random.uniform(1.1, 1.25)
        b_scale = random.uniform(1.1, 1.6)
        if testing:
            r_scale = 0.57
            g_scale = 1.125
            b_scale = 1.6
    elif dominant == 'red':
        r_scale = random.uniform(1.1, 1.3)
        g_scale = random.uniform(0.75, 1.0)
        b_scale = random.uniform(0.75, 1.0)
        if testing:
            r_scale = 1.25
            g_scale = 0.8
            b_scale = 0.8
    elif dominant == 'yellow':
        r_scale = random.uniform(0.85, 1.15)
        g_scale = random.uniform(0.85, 1.15)
        b_scale = random.uniform(1.0, 1.35)
        if testing:
            r_scale = 0.9
            g_scale = 0.9
            b_scale = 1.3
    elif dominant == 'green':
        r_scale = random.uniform(0.75, 1.0)
        g_scale = random.uniform(1.1, 1.3)
        b_scale = random.uniform(0.75, 1.0)
        if testing:
            r_scale = 0.8
            g_scale = 1.25
            b_scale = 0.8
    else:
        if testing:
            print(f"Unknown label: {label}. Using default scaling.")
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

    x1, x2 = sorted([random.uniform(0.1, 0.4), random.uniform(0.6, 0.9)])
    y1, y2 = sorted([random.uniform(0.1, 0.4), random.uniform(0.3, 0.5)])

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
    aug = A.Compose([
        RandomRain(
            blur_value=random.choice([2, 3]),
            brightness_coefficient=random.uniform(0.753, 0.95),
            always_apply=True
        )
    ])
    return aug(image=image)['image']

# def apply_albumentations_enhancements(image):
#     # Random brightness, contrast, hue, saturation, and value adjustments
#     aug = A.Compose([
#         RandomBrightnessContrast(
#             brightness_limit=0.15, contrast_limit=0.23, p=1.0),
#         HueSaturationValue(
#             hue_shift_limit=15, sat_shift_limit=15, val_shift_limit=15, p=1.0)
#     ])
#     return aug(image=image)['image']

def flip_lr(image):
    # Horizontal flip
    return cv2.flip(image, 1)

def rotate(image):
    h, w = image.shape[:2]
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def perspective_warp(image):
    h, w = image.shape[:2]
    max_offset = int(w * 0.5 / 2)
    offset = random.randint(int(max_offset * 0.3), max_offset)
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([
        [offset, 0],
        [w - offset, 0],
        [offset, h],
        [w - offset, h]
    ])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(image, matrix, (w, h))
    x_min = int(min(pts2[:, 0]))
    x_max = int(max(pts2[:, 0]))
    y_min = int(min(pts2[:, 1]))
    y_max = int(max(pts2[:, 1]))
    cropped = warped[y_min:y_max, x_min:x_max]
    return cropped

# -----------------------------
# Main for Testing Augmentations
# -----------------------------

if __name__ == "__main__":
    image_path = os.path.join(repo_path, "bfmc_data", "base", "crop", "oneway", "63.jpg")
    image_path = os.path.join(repo_path, "bfmc_data", "base", "crop", "crosswalk", "20.JPG")
    image_path = os.path.join(repo_path, "bfmc_data", "base", "crop", "crosswalk", "28.jpg")
    image_paths = [
        # size
        # os.path.join(repo_path, "bfmc_data", "base", "crop", "crosswalk", "28.jpg"),
        # os.path.join(repo_path, "bfmc_data", "base", "crop", "crosswalk", "20.JPG"),
        
        # color
        # os.path.join(repo_path, "bfmc_data", "base", "crop", "crosswalk", "frame_531_5_269_0.jpg"), #lightest
        # os.path.join(repo_path, "bfmc_data", "base", "crop", "crosswalk", "22.jpg"), # darkest
        # os.path.join(repo_path, "bfmc_data", "base", "crop", "crosswalk", "175.jpg"), # darkest
        # os.path.join(repo_path, "bfmc_data", "base", "crop", "crosswalk", "20.JPG"), # default
        # os.path.join(repo_path, "bfmc_data", "base", "crop", "crosswalk", "frame_669_5_535_80.jpg"), #normal
        # os.path.join(repo_path, "bfmc_data", "base", "crop", "crosswalk", "44.jpg"), #normal
        
        # all signs
        os.path.join(repo_path, "bfmc_data", "base", "crop", "crosswalk", "44.jpg"), #normal
        os.path.join(repo_path, "bfmc_data", "base", "crop", "crosswalk", "frame_669_5_535_80.jpg"), #normal
        os.path.join(repo_path, "bfmc_data", "base", "crop", "highwayentrance", "frame_123_1_150_86.jpg"), #normal
        os.path.join(repo_path, "bfmc_data", "base", "crop", "highwayentrance", "frame_1743441347_1_148_54.jpg"), #normal
        os.path.join(repo_path, "bfmc_data", "base", "crop", "noentry", "frame_635_6_68_270.jpg"), #normal
        os.path.join(repo_path, "bfmc_data", "base", "crop", "noentry", "frame_1743441347_6_0_237.jpg"), #normal
        os.path.join(repo_path, "bfmc_data", "base", "crop", "prio", "frame_663_8_531_191.jpg"), #normal
        os.path.join(repo_path, "bfmc_data", "base", "crop", "prio", "234.jpg"), #normal
        os.path.join(repo_path, "bfmc_data", "base", "crop", "stopsign", "217.jpg"), #normal
        os.path.join(repo_path, "bfmc_data", "base", "crop", "stopsign", "frame_619_2_233_144.jpg"), #normal
        os.path.join(repo_path, "bfmc_data", "base", "crop", "stopsign", "frame_713_2_558_87.jpg"), #normal
        os.path.join(repo_path, "bfmc_data", "base", "crop", "stopsign", "282.jpg"), #normal
        
    ]
    # for image_path in image_paths:
    for image_path, i in zip(image_paths, range(len(image_paths))):
        print(f"{i}) {image_path}")
        image = cv2.imread(image_path)
        cv2.imshow('Original' + str(i), image)
        
        #size dependent
        # cv2.imshow(f'Motion Blur {i}', apply_motion_blur(image.copy(), testing=True))
        # cv2.imshow(f'Pixelated {i}', apply_pixelation(image.copy(), testing=True))
        # cv2.imshow(f'Defocus Blur {i}', apply_defocus_blur(image.copy(), testing=True))
        # cv2.imshow(f'Sun {i}', apply_sun(image.copy()))
        
        #color dependent
        # cv2.imshow(f'Brightness increased {i}', increase_brightness(image.copy(), testing=True))
        # cv2.imshow(f'Brightness decreased {i}', decrease_brightness(image.copy(), testing=True))
        # cv2.imshow(f'Contrast Increased {i}', increase_contrast(image.copy()))
        # cv2.imshow(f'Contrast Decreased {i}', decrease_contrast(image.copy()))
        # cv2.imshow(f'Saturation Increased {i}', increase_saturation(image.copy(), testing=True))
        # cv2.imshow(f'Saturation Decreased {i}', decrease_saturation(image.copy(), testing=True))
        # cv2.imshow(f'Color Temperature {i}', apply_color_temperature(image.copy(), testing=True))
        # cv2.imshow(f'Strong Color Shift {i}', strong_color_shift(image.copy(), image_path, testing=True))
        # cv2.imshow(f'Rain {i}', apply_rain(image.copy()))
        
        # doesnt matter
        # cv2.imshow(f'Flip LR {i}', flip_lr(image.copy()))
        # cv2.imshow(f'Rotated {i}', rotate(image.copy()))
        # cv2.imshow(f'Perspective Warp {i}', perspective_warp(image.copy()))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
