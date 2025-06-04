import os
import sys
sys.path.append("/home/nafisur/Documents/VASCO/custom_yolo/ultralytics")  # Adjust path to import from parent directory
import argparse
import yaml
from ultralytics.data.dataset import YOLODataset  # Adjust path to import from parent directory

img_path = '/home/nafisur/Documents/VASCO/processed_multi/images/train'
batch = 1
mode = "train"
# cfg = "config"
stride = 32

# read params from a JSON file
params_file = '/home/nafisur/Documents/VASCO/custom_yolo/cfg.yaml'
with open(params_file) as f:
    params = yaml.safe_load(f)

params = argparse.Namespace(**params)  # Convert dict to Namespace for compatibility with YOLODataset
# print(params)

data = {'path': '/home/nafisur/Documents/VASCO/processed_detection/det_MY2', 'val': '/home/nafisur/Documents/VASCO/processed_detection/det_MY2/images/val', 'train': '/home/nafisur/Documents/VASCO/processed_detection/det_MY2/images/train', 'nc': 14, 'names': {0: 'person', 1: 'car', 2: 'bus', 3: 'van', 4: 'truck', 5: 'motorcycle', 6: 'bicycle', 7: 'scooter', 8: 'trash_bin', 9: 'pole', 10: 'construction_barriers', 11: 'red_light_p', 12: 'green_light_p', 13: 'orange_light_p'}, 'yaml_file': '/home/nafisur/Documents/VASCO/seg_pedestrian/data_det.yaml', 'channels': 3}
dataset = YOLODataset(img_path=img_path,
        imgsz=640,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=params,  # TODO: probably add a get_hyps_from_cfg function
        rect=False,  # rectangular batches
        cache=False,
        single_cls= False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=None,
        task='detect',
        classes=None,
        data=data,
        fraction= 1.0,
    )

for i, batch in enumerate(dataset):
    print(batch)
