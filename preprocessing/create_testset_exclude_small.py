#!/usr/bin/env python3
"""
Filter out objects whose bounding‑box width is below SMALL_WIDTH_PX.
Small boxes are masked (black or random colour) and removed from
the YOLO label file.

The script is multi‑process: each image/label pair is handled in a
separate worker to utilise multiple CPU cores.

Run with:  python filter_small_objects_mp.py
"""

from __future__ import annotations 
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

# ───────────────────────────────────────────────────────────
# CONFIGURATION  –– ADJUST TO YOUR SETUP
# ───────────────────────────────────────────────────────────
repo_path = Path(__file__).resolve().parent.parent   # repo root

random.seed(357)
np.random.seed(357)

NAME = "TestSet0616"  # output name
SRC_ROOT          = repo_path / "bfmc_data" / "generated" / "testsets" / NAME
DST_ROOT          = repo_path / "bfmc_data" / "generated" / "testsets" / f"{NAME}_filtered"
SMALL_WIDTH_PX    = 10     # mask + drop any object narrower than this
USE_RANDOM_COLOR  = True   # True ⇒ random RGB; False ⇒ solid black
NUM_WORKERS       = os.cpu_count() or 4
# ───────────────────────────────────────────────────────────


# ---------- utility helpers ----------
def ensure_dirs(dst: Path) -> None:
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)


def find_corresponding_image(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def mask_and_filter(
    img: np.ndarray,
    label_lines: list[str],
    img_w: int,
    img_h: int,
    small_width_px: int,
    use_random: bool,
) -> tuple[np.ndarray, list[str]]:
    kept = []
    for line in label_lines:
        cls, x_c, y_c, bw, bh = map(float, line.split())
        box_px_w = bw * img_w

        # YOLO → absolute pixel coords
        x1 = int((x_c - bw / 2) * img_w)
        y1 = int((y_c - bh / 2) * img_h)
        x2 = int((x_c + bw / 2) * img_w)
        y2 = int((y_c + bh / 2) * img_h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)

        if box_px_w < small_width_px:
            colour = (
                (0, 0, 0)
                if not use_random
                else tuple(int(random.randrange(256)) for _ in range(3))
            )
            img[y1:y2, x1:x2] = colour
        else:
            kept.append(line)
    return img, kept


# ---------- worker function ----------
def process_one(
    lbl_path: str,
    src_images_dir: str,
    dst_images_dir: str,
    dst_labels_dir: str,
    small_width_px: int,
    use_random_colour: bool,
) -> tuple[str, int, int]:
    """
    Worker executed in a separate process.

    Returns (filename, kept_count, masked_count)
    """
    # Re‑seed per process for varied random colours
    random.seed(os.getpid() + 489)
    np.random.seed(os.getpid() + 489)

    lbl_path = Path(lbl_path)
    stem = lbl_path.stem
    img_path = find_corresponding_image(Path(src_images_dir), stem)
    if img_path is None:
        return (lbl_path.name, 0, 0)  # will be reported as warning later

    img = cv2.imread(str(img_path))
    if img is None:
        return (lbl_path.name, 0, 0)

    h, w = img.shape[:2]
    with open(lbl_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    img, kept_lines = mask_and_filter(
        img,
        lines,
        w,
        h,
        small_width_px=small_width_px,
        use_random=use_random_colour,
    )

    # save outputs
    cv2.imwrite(str(Path(dst_images_dir) / img_path.name), img)
    with open(Path(dst_labels_dir) / lbl_path.name, "w") as f:
        f.write("\n".join(kept_lines))

    return (img_path.name, len(kept_lines), len(lines) - len(kept_lines))


# ---------- main driver ----------
def main() -> None:
    ensure_dirs(DST_ROOT)

    src_images_dir = SRC_ROOT / "images"
    src_labels_dir = SRC_ROOT / "labels"
    dst_images_dir = DST_ROOT / "images"
    dst_labels_dir = DST_ROOT / "labels"

    label_files = sorted(src_labels_dir.glob("*.txt"))
    if not label_files:
        print("No label files found in", src_labels_dir)
        sys.exit(1)

    print(f"Processing {len(label_files)} files with {NUM_WORKERS} workers …\n")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = [
            pool.submit(
                process_one,
                str(lbl),
                str(src_images_dir),
                str(dst_images_dir),
                str(dst_labels_dir),
                SMALL_WIDTH_PX,
                USE_RANDOM_COLOR,
            )
            for lbl in label_files
        ]

        for fut in as_completed(futures):
            fname, keep, mask = fut.result()
            if (keep + mask) == 0:
                print(f"[WARN] {fname} – image missing or unreadable")
            else:
                print(f"{fname:<30} kept {keep:>2} • masked {mask:>2}")

    print("\nDone!  Filtered test set written to:", DST_ROOT)


if __name__ == "__main__":
    main()
