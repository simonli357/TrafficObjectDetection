import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backend_bases import MouseButton
import shutil
import threading
import time
import numpy as np
from pathlib import Path
import random

random.seed(357)

repo_path = Path(__file__).resolve().parent.parent

class_names = [
    "oneway", "highwayentrance", "stopsign", "roundabout", "park",
    "crosswalk", "noentry", "highwayexit", "prio", "light",
    "roadblock", "girl", "cars2"
]

matplotlib.rcParams['keymap.save'].remove('s')
for k in ['a', 'p', 'o', 'i', 'x', 'n', 'b', 'q', 'left', 'right', 'up', 'down']:
    if k in matplotlib.rcParams['keymap.pan']:
        matplotlib.rcParams['keymap.pan'].remove(k)
    if k in matplotlib.rcParams['keymap.zoom']:
        matplotlib.rcParams['keymap.zoom'].remove(k)
# ---------------- Configuration ----------------
# root = os.path.join(repo_path, "bfmc_data", "base", "testsets", "xinya")
# root = os.path.join(repo_path, "bfmc_data", "base", "datasets", "datasets_g")
root = os.path.join(repo_path, "bfmc_data", "base", "unprocessed", "frames_0402_lights")
root = os.path.join(repo_path, "bfmc_data", "base", "datasets", "datasets_city_padded")
# root = os.path.join(repo_path, "bfmc_data", "base", "datasets", "datasets_a")
DEBUG = False

IMAGE_FOLDER = os.path.join(root, "images")
LABEL_FOLDER = os.path.join(root, "labels")
TRASH_FOLDER = os.path.join(root, "_trash")
os.makedirs(LABEL_FOLDER, exist_ok=True)
BACKGROUND_FOLDER = repo_path / "bfmc_data" / "base" / "backgrounds"

CLASS_NAMES = [
    "oneway", "highwayentrance", "stop", "roundabout", "parking",
    "crosswalk", "noentry", "highwayexit", "prio", "light",
    "block", "girl", "car"
]
CLASS_COLORS = ['g', 'b', 'r', 'c', 'm', 'y', 'lime', 'orange', 'olive', 'teal', 'maroon', 'navy', 'purple']
CLASS_KEYS = {str(i): i for i in range(10)}
CLASS_KEYS.update({"i": 10, "o": 11, "p": 12})
HANDLE_RADIUS = 5

# ---------------- Annotation Tool Class ----------------
class BoundingBox:
    def __init__(self, class_id, x, y, w, h, confidence=1.0):  # ← add confidence default
        self.class_id = class_id
        self.x, self.y, self.w, self.h = x, y, w, h
        self.confidence = confidence  # ← store it
        self.selected = False
        self.handle_selected = None

    def clamp_to_bounds(self, img_w, img_h):
        # Ensure width and height are non-negative
        self.w = max(1, self.w)
        self.h = max(1, self.h)

        # Clamp x/y to image borders
        if self.x < 0:
            self.w += self.x  # reduce width
            self.x = 0
        if self.y < 0:
            self.h += self.y  # reduce height
            self.y = 0

        # Prevent x+w or y+h from going out of bounds
        if self.x + self.w > img_w:
            self.w = img_w - self.x
        if self.y + self.h > img_h:
            self.h = img_h - self.y

        # Final safety clamp
        self.w = max(1, self.w)
        self.h = max(1, self.h)
    def get_rect(self):
        return self.x, self.y, self.w, self.h

    def get_handles(self):
        x, y, w, h = self.get_rect()
        return {
            'tl': (x, y),
            'tr': (x + w, y),
            'bl': (x, y + h),
            'br': (x + w, y + h),
            't': (x + w // 2, y),
            'b': (x + w // 2, y + h),
            'l': (x, y + h // 2),
            'r': (x + w, y + h // 2)
        }

    def draw(self, ax):
        color = 'white' if self.selected else CLASS_COLORS[self.class_id % len(CLASS_COLORS)]
        if DEBUG:
            linewidth = 6
            fontsize = 16
            weight = 'bold'
        else:
            linewidth = 2
            fontsize = 8
            weight = 'normal'
        rect = patches.Rectangle((self.x, self.y), self.w, self.h, linewidth=linewidth, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        label = f"{CLASS_NAMES[self.class_id]} ({self.confidence:.2f})"
        ax.text(self.x, self.y - 5, label, color=color, fontsize=fontsize, weight=weight)

        if self.selected:
            # Get dynamic radius based on zoom level
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            view_width = abs(xlim[1] - xlim[0])
            view_height = abs(ylim[1] - ylim[0])
            base_size = (view_width + view_height) / 2 * 4
            dynamic_radius = max(base_size * 0.003, 2)  # min radius 2

            for _, (hx, hy) in self.get_handles().items():
                handle = patches.Circle((hx, hy), radius=dynamic_radius, color=color)
                ax.add_patch(handle)

    def contains(self, x, y):
        return self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h

    def handle_hit(self, x, y):
        for name, (hx, hy) in self.get_handles().items():
            if abs(x - hx) <= HANDLE_RADIUS and abs(y - hy) <= HANDLE_RADIUS:
                return name
        return None
    def resize(self, handle, dx, dy):
        if handle == 'tl':
            self.x += dx; self.y += dy; self.w -= dx; self.h -= dy
        elif handle == 'tr':
            self.y += dy; self.w += dx; self.h -= dy
        elif handle == 'bl':
            self.x += dx; self.w -= dx; self.h += dy
        elif handle == 'br':
            self.w += dx; self.h += dy
        elif handle == 't':
            self.y += dy; self.h -= dy
        elif handle == 'b':
            self.h += dy
        elif handle == 'l':
            self.x += dx; self.w -= dx
        elif handle == 'r':
            self.w += dx

class AnnotationApp:
    def __init__(self):
        import re
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
        self.image_files = sorted(
            [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg", ".png"))],
            key=natural_sort_key
        )
        self.index = 0
        self.boxes = []
        self.selected_box = None
        self.adding_box = False
        self.start = None
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.current_class = 0
        self.xlim = None
        self.ylim = None
        self.dragging = False
        self.last_mouse_pos = None

        self.fig, self.ax = plt.subplots()
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_scroll = self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.load_image()
        plt.show()

    def load_image(self):
        self.zoom = 1.0
        self.boxes.clear()
        img_path = os.path.join(IMAGE_FOLDER, self.image_files[self.index])
        self.image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        self.img_h, self.img_w = self.image.shape[:2]
        label_path = os.path.join(LABEL_FOLDER, os.path.splitext(self.image_files[self.index])[0] + ".txt")
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    result = self.yolo_to_box(line)
                    if result is not None:
                        cid, x, y, w, h, conf = result
                        self.boxes.append(BoundingBox(cid, x, y, w, h, confidence=conf))
        self.reset_view()
        self.redraw()

    def reset_view(self):
        self.xlim = (0, self.img_w)
        self.ylim = (self.img_h, 0)

    def yolo_to_box(self, line, conf_thresh=0.3):
        parts = line.strip().split()
        if len(parts) < 5:
            return None  # invalid line

        cid = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])

        # Handle confidence score if present
        conf = float(parts[5]) if len(parts) >= 6 else 1.0
        if conf < conf_thresh:
            return None

        x = (xc - w / 2) * self.img_w
        y = (yc - h / 2) * self.img_h
        w_pixels = w * self.img_w
        h_pixels = h * self.img_h
        return cid, x, y, w_pixels, h_pixels, conf

    def box_to_yolo(self, box):
        xc = (box.x + box.w / 2) / self.img_w
        yc = (box.y + box.h / 2) / self.img_h
        w = box.w / self.img_w
        h = box.h / self.img_h
        
        return f"{box.class_id} {xc:.8f} {yc:.8f} {w:.8f} {h:.8f}"

    def redraw(self):
        self.ax.cla()  # even better than .clear()
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        self.ax.imshow(self.image, extent=(0, self.img_w, self.img_h, 0))
        for box in self.boxes:
            box.draw(self.ax)
        self.ax.set_title(f"{self.image_files[self.index]} | Class: {CLASS_NAMES[self.current_class]} | Zoom: {self.zoom:.2f}")
        self.fig.canvas.draw_idle()  # better performance than draw()

    def on_press(self, event):
        if event.inaxes != self.ax or event.button != MouseButton.LEFT:
            return

        self.last_mouse_pos = (event.xdata, event.ydata)
        
        if self.adding_box:
            self.start = (event.xdata, event.ydata)
            self.last_mouse_pos = None
            return

        # Check if a handle is clicked
        for box in self.boxes:
            if box.selected:
                handle = box.handle_hit(event.xdata, event.ydata)
                if handle:
                    box.handle_selected = handle
                    self.selected_box = box
                    self.dragging = True
                    return

        # Check if inside a box for moving
        for box in reversed(self.boxes):
            if box.contains(event.xdata, event.ydata):
                for b in self.boxes:
                    b.selected = False
                box.selected = True
                box.handle_selected = 'move'
                self.selected_box = box
                self.dragging = True
                self.redraw()
                return

        # Otherwise, deselect all
        for b in self.boxes:
            b.selected = False
        self.selected_box = None
        self.redraw()

    def on_motion(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        if self.adding_box and self.start:
            self.redraw()
            if event.xdata is not None and event.ydata is not None:
                # Draw crosshair lines
                self.ax.axhline(event.ydata, color='cyan', linestyle='--', linewidth=0.5)
                self.ax.axvline(event.xdata, color='cyan', linestyle='--', linewidth=0.5)
            # If dragging to draw box, draw preview box too
            if self.start:
                x0, y0 = self.start
                x1, y1 = event.xdata, event.ydata
                self.ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                                    linewidth=1, edgecolor='cyan', linestyle='--', facecolor='none'))

            self.fig.canvas.draw()
            return
        if not self.dragging or not self.selected_box or event.inaxes != self.ax or self.last_mouse_pos is None:
            return

        dx = event.xdata - self.last_mouse_pos[0]
        dy = event.ydata - self.last_mouse_pos[1]

        if self.selected_box.handle_selected == 'move':
            self.selected_box.x += dx
            self.selected_box.y += dy
        else:
            self.selected_box.resize(self.selected_box.handle_selected, dx, dy)

        self.last_mouse_pos = (event.xdata, event.ydata)
        self.redraw()

    def on_release(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        if self.adding_box and self.start:
            x0, y0 = self.start
            x1, y1 = event.xdata, event.ydata
            x, y = min(x0, x1), min(y0, y1)
            w, h = abs(x1 - x0), abs(y1 - y0)
            if w > 5 and h > 5:
                self.boxes.append(BoundingBox(self.current_class, x, y, w, h))
            self.adding_box = False
            self.start = None
            self.redraw()
            return
        
        self.dragging = False
        self.last_mouse_pos = None
        if self.selected_box:
            self.selected_box.handle_selected = None

    def on_scroll(self, event):
        factor = 1.1 if event.button == 'up' else 0.9
        x_center = event.xdata
        y_center = event.ydata
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        new_width = (x1 - x0) / factor
        new_height = (y0 - y1) / factor
        self.xlim = (x_center - new_width / 2, x_center + new_width / 2)
        self.ylim = (y_center + new_height / 2, y_center - new_height / 2)
        self.zoom *= factor
        self.redraw()

    def on_key(self, event):
        if event.key in CLASS_KEYS:
            new_class = CLASS_KEYS[event.key]
            if self.selected_box:
                self.selected_box.class_id = new_class
                print(f"Changed selected box to class: {CLASS_NAMES[new_class]}")
            else:
                self.current_class = new_class
            self.redraw()
        elif event.key == 'a':
            self.adding_box = True
        elif event.key == 'r':
            if self.boxes:
                self.boxes.pop()
                self.redraw()
        elif event.key == 'd' and self.selected_box:
            self.boxes.remove(self.selected_box)
            self.selected_box = None
            self.redraw()
        elif event.key == 's':
            for box in self.boxes:
                box.clamp_to_bounds(self.img_w, self.img_h)
            label_path = os.path.join(LABEL_FOLDER, os.path.splitext(self.image_files[self.index])[0] + ".txt")
            with open(label_path, 'w') as f:
                for box in self.boxes:
                    f.write(self.box_to_yolo(box) + "\n")
            img_path = os.path.join(IMAGE_FOLDER, self.image_files[self.index])
            cv2.imwrite(img_path, cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
            print("Saved")
            
            original_title = self.ax.get_title()
            self.ax.set_title(f"{original_title} — ✅ Saved")
            self.fig.canvas.draw()

            # Revert title after short delay
            def reset_title():
                time.sleep(1.5)
                self.ax.set_title(original_title)
                self.fig.canvas.draw()
            threading.Thread(target=reset_title, daemon=True).start()
        elif event.key == 'x':
            # Move image and label to trash
            img_path = os.path.join(IMAGE_FOLDER, self.image_files[self.index])
            label_path = os.path.join(LABEL_FOLDER, os.path.splitext(self.image_files[self.index])[0] + ".txt")
            os.makedirs(TRASH_FOLDER, exist_ok=True)
            
            shutil.move(img_path, os.path.join(TRASH_FOLDER, os.path.basename(img_path)))
            
            if os.path.exists(label_path):
                shutil.move(label_path, os.path.join(TRASH_FOLDER, os.path.basename(label_path)))
            
            print(f"Moved {self.image_files[self.index]} and label to trash.")

            self.image_files.pop(self.index)
            
            if self.index >= len(self.image_files) and self.image_files:
                self.index = len(self.image_files) - 1
            
            if self.image_files:
                self.load_image()
            else:
                print("Done")
                plt.close()
        elif event.key == 'n':
            # while self.index + 1 < len(self.image_files):
            #     accepted_indices = [9]
            #     # accepted_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]
            #     self.index += 1
            #     label_path = os.path.join(LABEL_FOLDER, os.path.splitext(self.image_files[self.index])[0] + ".txt")
            #     if os.path.exists(label_path):
            #         with open(label_path) as f:
            #             for line in f:
            #                 parts = line.strip().split()
            #                 if len(parts) >= 1 and int(float(parts[0])) in accepted_indices:
            #                     self.load_image()
            #                     return
            # print("No more images with class 9.")
            # plt.close()
            self.index += 1
            if self.index < len(self.image_files):
                self.load_image()
            else:
                print("Done")
                plt.close()
        elif event.key == 'b':
            if self.index > 0:
                self.index -= 1
                self.load_image()
        elif event.key == 'q':
            plt.close()
        elif event.key == 'h' and self.selected_box:
            box = self.selected_box
            x1 = int(max(0, round(box.x)))
            y1 = int(max(0, round(box.y)))
            x2 = int(min(self.img_w, round(box.x + box.w)))
            y2 = int(min(self.img_h, round(box.y + box.h)))
            bw = x2 - x1
            bh = y2 - y1

            if bw > 0 and bh > 0:
                # Load a random background image
                bg_files = [f for f in os.listdir(BACKGROUND_FOLDER) if f.lower().endswith(('.jpg', '.png'))]
                if not bg_files:
                    print("No background images found.")
                    return
                bg_path = os.path.join(BACKGROUND_FOLDER, random.choice(bg_files))
                bg_img = cv2.cvtColor(cv2.imread(bg_path), cv2.COLOR_BGR2RGB)
                bh_img, bw_img = bg_img.shape[:2]

                # If bg image too small, resize it to fit at least the patch size
                if bw_img < bw or bh_img < bh:
                    scale_w = bw / bw_img
                    scale_h = bh / bh_img
                    scale = max(scale_w, scale_h)
                    bg_img = cv2.resize(bg_img, (int(bw_img * scale), int(bh_img * scale)))
                    bh_img, bw_img = bg_img.shape[:2]

                # Choose random top-left position
                max_x = bw_img - bw
                max_y = bh_img - bh
                start_x = random.randint(0, max_x)
                start_y = random.randint(0, max_y)

                patch = bg_img[start_y:start_y+bh, start_x:start_x+bw]

                self.image[y1:y2, x1:x2] = patch
                self.redraw()

                print(f"Replaced region ({x1},{y1})–({x2},{y2}) with patch from {os.path.basename(bg_path)}")
        elif event.key == 'm' and self.selected_box:
            box = self.selected_box
            x1 = int(max(0, round(box.x)))
            y1 = int(max(0, round(box.y)))
            x2 = int(min(self.img_w, round(box.x + box.w)))
            y2 = int(min(self.img_h, round(box.y + box.h)))

            if x2 > x1 and y2 > y1:
                # Generate random noise the same shape as the region
                random_pixels = (np.random.rand(y2 - y1, x2 - x1, 3) * 255).astype(np.uint8)
                self.image[y1:y2, x1:x2] = random_pixels

                self.redraw()

                print(f"Masked region: ({x1},{y1}) to ({x2},{y2}) with random pixels.")
        elif event.key == 'c':
            for box in self.boxes:
                class_dir = os.path.join(root, str(class_names[int(box.class_id)]))
                os.makedirs(class_dir, exist_ok=True)

                x1 = int(max(0, round(box.x)))
                y1 = int(max(0, round(box.y)))
                x2 = int(min(self.img_w, round(box.x + box.w)))
                y2 = int(min(self.img_h, round(box.y + box.h)))

                cropped = self.image[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue  # Skip invalid crops

                base_name = os.path.splitext(self.image_files[self.index])[0]
                crop_name = f"{base_name}_{box.class_id}_{x1}_{y1}.jpg"
                crop_path = os.path.join(class_dir, crop_name)

                cv2.imwrite(crop_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
            original_title = self.ax.get_title()
            self.ax.set_title(f"{original_title} — ✅ Saved Cropped")
            self.fig.canvas.draw()

            # Revert title after short delay
            def reset_title():
                time.sleep(1.5)
                self.ax.set_title(original_title)
                self.fig.canvas.draw()
            threading.Thread(target=reset_title, daemon=True).start()
            print("Cropped images saved as ", crop_path)
        elif event.key == 'left':
            dx = 20
            self.xlim = (self.xlim[0] - dx, self.xlim[1] - dx)
            self.redraw()
        elif event.key == 'right':
            dx = 20
            self.xlim = (self.xlim[0] + dx, self.xlim[1] + dx)
            self.redraw()
        elif event.key == 'up':
            dy = 20
            self.ylim = (self.ylim[0] - dy, self.ylim[1] - dy)
            self.redraw()
        elif event.key == 'down':
            dy = 20
            self.ylim = (self.ylim[0] + dy, self.ylim[1] + dy)
            self.redraw()

if __name__ == "__main__":
    AnnotationApp()
