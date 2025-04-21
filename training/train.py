import os
import yaml
from ultralytics import YOLO
import shutil
from pathlib import Path

repo_path = Path(__file__).resolve().parent.parent
NAME = 'core0421'
num_epochs = 30
DATASET_NAME = 'datasets_0421'
results_dir = os.path.join(repo_path, 'training', 'runs')
augment_path = os.path.join(repo_path, 'config/augment_config_default.yaml')
model_path = os.path.join(repo_path, 'training', 'models', 'yolov8n.pt')
data={
    'train': os.path.join(repo_path, 'bfmc_data','generated', DATASET_NAME),
    'val': os.path.join(repo_path, 'bfmc_data','generated','testsets','TestSetAll'),
    'test': os.path.join(repo_path, 'bfmc_data','generated','testsets','TestSetAll'),
    'num_epochs': num_epochs,
    'name': NAME,
    'names': {
        0: 'oneway',
        1: 'highwayentrance',
        2: 'stop',
        3: 'roundabout',
        4: 'parking',
        5: 'crosswalk',
        6: 'noentry',
        7: 'highwayexit',
        8: 'prio',
        9: 'trafficlight',
        10: 'block',
        11: 'girl',
        12: 'car'
    }
}
generated_yaml_path = os.path.join(repo_path, 'config', 'train_config.yaml')
with open(generated_yaml_path, 'w') as f:
    yaml.dump(data, f)
yaml_path = os.path.join(repo_path, 'config/train_config.yaml')

with open(yaml_path, 'r') as f:
    data_config = yaml.safe_load(f)
with open(augment_path, 'r') as f:
    augment_config = yaml.safe_load(f)

model = YOLO(model_path)
print(f"🚀 Model is using device: {model.device}")

output_dir = os.path.join(results_dir, NAME + str(num_epochs))
os.makedirs(os.path.join(output_dir, 'config'), exist_ok=True)
shutil.copy2(yaml_path, os.path.join(output_dir, 'config', 'train_config.yaml'))
shutil.copy2(augment_path, os.path.join(output_dir, 'config', 'augment_config.yaml'))
local_augment_path = repo_path / 'preprocessing' / 'augmentations.py'
local_augment_path2 = repo_path / 'preprocessing' / 'apply_augmentations.py'
resize_path = repo_path / 'preprocessing' / 'resize_normal.py'
combine_datasets_path = repo_path / 'preprocessing' / 'combine_datasets.py'
shutil.copy2(local_augment_path, os.path.join(output_dir, 'config', 'augmentations.py'))
shutil.copy2(local_augment_path2, os.path.join(output_dir, 'config', 'apply_augmentations.py'))
shutil.copy2(resize_path, os.path.join(output_dir, 'config', 'resize_normal.py'))
shutil.copy2(combine_datasets_path, os.path.join(output_dir, 'config', 'combine_datasets.py'))

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

print(f"\n✅ Training complete. Results saved in: {results_dir}")
