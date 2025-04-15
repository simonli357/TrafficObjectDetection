import os

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

LABEL_FOLDER = os.path.join(repo_path, "bfmc_data", "base", "datasets", "datasets_bb", "labels")
OUTPUT_FOLDER = os.path.join(repo_path, "bfmc_data", "base", "datasets", "datasets_bb", "labels_adjusted")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Adjustments in pixels
SHIFT_X = 0.       # horizontal shift (+right, −left)
SHIFT_Y = 0.       # vertical shift (+down, −up)
DELTA_W = -1       # change in width (+bigger, −smaller)
DELTA_H = -1       # change in height

def adjust_box(xc, yc, w, h):
    # Convert from YOLO-normalized to pixel
    px = xc * IMAGE_WIDTH
    py = yc * IMAGE_HEIGHT
    pw = w * IMAGE_WIDTH
    ph = h * IMAGE_HEIGHT

    # Apply shifts
    px += SHIFT_X
    py += SHIFT_Y
    pw += DELTA_W
    ph += DELTA_H

    # Clamp to positive width/height
    pw = max(pw, 1)
    ph = max(ph, 1)

    # Convert back to YOLO-normalized
    xc_new = px / IMAGE_WIDTH
    yc_new = py / IMAGE_HEIGHT
    w_new = pw / IMAGE_WIDTH
    h_new = ph / IMAGE_HEIGHT
    return xc_new, yc_new, w_new, h_new

for fname in os.listdir(LABEL_FOLDER):
    if not fname.endswith(".txt"):
        continue

    input_path = os.path.join(LABEL_FOLDER, fname)
    output_path = os.path.join(OUTPUT_FOLDER, fname)

    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # skip malformed lines

            class_id = parts[0]
            xc, yc, w, h = map(float, parts[1:5])

            xc_adj, yc_adj, w_adj, h_adj = adjust_box(xc, yc, w, h)

            # Optional: clamp to [0,1] for YOLO format (uncomment if needed)
            # xc_adj = min(max(xc_adj, 0), 1)
            # yc_adj = min(max(yc_adj, 0), 1)
            # w_adj = min(max(w_adj, 0), 1)
            # h_adj = min(max(h_adj, 0), 1)

            # Write adjusted line
            f_out.write(f"{class_id} {xc_adj:.6f} {yc_adj:.6f} {w_adj:.6f} {h_adj:.6f}\n")

print("✅ Done adjusting boxes.")
