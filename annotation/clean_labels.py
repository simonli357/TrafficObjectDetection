import os
from pathlib import Path

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def clamp(val, min_val=0.0, max_val=1.0):
    return max(min(val, max_val), min_val)

def clean_and_fix_labels(label_dir):
    label_dir = Path(label_dir)
    label_files = list(label_dir.glob("*.txt"))
    fixed_count = 0

    for file in label_files:
        fixed_lines = []
        changed = False

        with open(file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue  # skip malformed lines
                try:
                    cls = int(parts[0])
                    x, y, w, h = map(float, parts[1:])

                    # Fix values
                    x = clamp(x)
                    y = clamp(y)
                    w = abs(w)
                    h = abs(h)

                    # Skip zero-sized boxes
                    if w == 0 or h == 0:
                        changed = True
                        continue

                    fixed_lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                except ValueError:
                    print("error: ", file)
                    continue

        if fixed_lines:
            with open(file, "w") as f:
                f.write("\n".join(fixed_lines) + "\n")
            if changed:
                fixed_count += 1
        else:
            file.unlink()  # Delete empty label file

    print(f"✅ Fixed {fixed_count} label files with out-of-bound or negative values.")
    print("🧹 Empty label files were deleted.")


clean_and_fix_labels(os.path.join(repo_path, "bfmc_data", "base", "datasets", "datasets_g", "labels"))
