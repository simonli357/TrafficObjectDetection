import os
import yaml
from ultralytics import YOLO
import shutil
from pathlib import Path

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

yaml_path = os.path.join(repo_path, 'config/train_config.yaml')
results_dir = os.path.join(repo_path, 'training', 'runs')
augment_path = os.path.join(repo_path, 'config/augment_config_noflip.yaml')
model_path = os.path.join(repo_path,'models/yolov8n.pt')
NAME = 'core_allxd'
num_epochs = 13

with open(yaml_path, 'r') as f:
    data_config = yaml.safe_load(f)
with open(augment_path, 'r') as f:
    augment_config = yaml.safe_load(f)

model = YOLO(model_path)
print(f"🚀 Model is using device: {model.device}")

model.train(
    device=0,
    save_period=1,
    data=yaml_path,
    epochs=num_epochs,
    imgsz=640,
    batch=16,
    patience=30,
    project=results_dir,
    name= NAME + str(num_epochs),
    exist_ok=True,
    **augment_config
)
output_dir = os.path.join(results_dir, NAME + str(num_epochs))
os.makedirs(os.path.join(output_dir, 'config'))
shutil.copy2(yaml_path, os.path.join(output_dir, 'config', 'train_config.yaml'))
shutil.copy2(augment_path, os.path.join(output_dir, 'config', 'augment_config.yaml'))

print(f"\n✅ Training complete. Results saved in: {results_dir}")
