from ultralytics import YOLO
import os
import cv2
import tqdm

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

MODEL_PATH = os.path.join(repo_path, 'training', 'models', 'xenia.pt')
name = 'bfmc2020'
root = os.path.join(repo_path, 'bfmc_data', 'base', 'unprocessed')
INPUT_DIR = os.path.join(root, name, 'images')
OUTPUT_LABEL_DIR = os.path.join(root, name, 'labels')
CONF_THRESHOLD = 0.25  # Confidence threshold for filtering detections

# Create output label directory if it doesn't exist
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# Load YOLOv8 model
model = YOLO(MODEL_PATH)

# Get list of image files in the input directory
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(image_extensions)]

# Process each image
# for image_file in image_files:
for image_file in tqdm.tqdm(image_files
                            , desc="Processing images"
                            , unit="image"
                            , unit_scale=True
                            , unit_divisor=1024
                            , dynamic_ncols=True):
    image_path = os.path.join(INPUT_DIR, image_file)
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Run inference
    results = model(image_path)[0]

    # Prepare output label file
    label_filename = os.path.splitext(image_file)[0] + '.txt'
    label_path = os.path.join(OUTPUT_LABEL_DIR, label_filename)

    with open(label_path, 'w') as f:
        for box in results.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())

            if conf < CONF_THRESHOLD:
                continue

            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Convert to YOLO format
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            bbox_width = (x2 - x1) / width
            bbox_height = (y2 - y1) / height

            # Write to label file
            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

print(f"✅ Inference complete. YOLO-format labels saved in: {OUTPUT_LABEL_DIR}")
