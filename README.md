
# Traffic Object Detection for BFMC

This repository contains scripts to train an object detection model for the **Bosch Future Mobility Challenge (BFMC)**: [https://boschfuturemobility.com/](https://boschfuturemobility.com/)

The model can detect and classify the following:
- **Traffic signs**: oneway, stop, highway entrance/exit, roundabout, parking, crosswalk, no entry, priority
- **Cars**
- **Pedestrians**

It includes tools for annotation, label generation using an existing model, preprocessing, synthetic dataset generation, training, evaluation, and analysis.

---

## 🛠 Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Pipeline

1. **Clone the repository**
   ```bash
   cd ~
   git clone https://github.com/simonli357/TrafficObjectDetection.git
   ```

2. **Download the dataset**

   Download the bfmc_data.zip from this Google Drive link:  
   *(available soon)*

3. **Unzip and create a symbolic link to the dataset**

   **Linux:**
   ```bash
   ln -s /path/to/unzipped/dataset ~/TrafficObjectDetection/bfmc_data
   ```

   **Windows (PowerShell):**
   ```powershell
   New-Item -ItemType SymbolicLink -Path "C:\Users\YourUser\TrafficObjectDetection\bfmc_data" -Target "D:\Path\To\Unzipped\Dataset"
   ```

4. **Generate datasets A and C:**

   a) Generate augmented cropped images:
   ```bash
   python3 preprocessing/apply_augmentations.py
   ```
   *(Tune values in the script to your needs)*

   b) Resize the images with Gaussian distribution:
   ```bash
   python3 preprocessing/resize_normal.py
   ```

   c) Create dataset A:
   ```bash
   python3 preprocessing/create_datasets_a.py
   ```

   d) Create dataset C:
   ```bash
   python3 preprocessing/create_datasets_c.py
   ```

5. **Combine all datasets into one**
   ```bash
   python3 preprocessing/combine_datasets.py
   ```

6. **Train the model**
   ```bash
   python3 training/train.py
   ```

7. **Test all epochs and save results**
   ```bash
   python3 training/test_all_epochs.py
   ```

8. **Plot results for all epochs**
   ```bash
   python3 training/plot_all_epochs.py
   ```

---

## Annotation Tools

1. **Generate labels for annotation**
   ```bash
   python3 annotation/generate_labels.py
   ```
   *(Modify the script to point to the images you want to annotate)*

2. **Adjust labels manually**
   ```bash
   python3 annotation/correct_labels.py
   ```

   This will open a GUI window with the image and bounding boxes:
   - Move or resize bounding boxes by clicking on them
   - Add new box: press `a` and draw
   - Change label class:
     - `0` to `9` → respective classes
     - `i` → class 10
     - `o` → class 11
     - `p` → class 12
   - Save: `s`
   - Navigate: `b` (back), `n` (next)
   - Crop and save bounding boxes: `c`
   - Hide part of the image: `h` and draw a box
   - Delete image and label: `x`
   - Quit: `q`