import os
import json
import shutil
import random
import numpy as np

# CONFIG
val_ratio = 0.15
random.seed(42)

# Category Mapping
# DETECTION_CLASSES = {
#     'bicycle': 0, 'bus': 1, 'car': 2, 'motorcycle': 3,
#     'person': 4, 'rider': 5, 'train': 6, 'truck': 7
# }
DETECTION_CLASSES = {
    "person": 0,
    "car": 1,
    "bus": 2,
    "van": 3,
    "truck": 4,
    "motorcycle": 5,
    "bicycle": 6,
    "scooter": 7,
    "trash_bin": 8,
    "pole": 9,
    "construction_barriers": 10,
    "red_light_p": 11,
    "green_light_p": 12,
    "orange_light_p": 13,
}

SEGMENTATION_CLASSES = {
    'bike_lane': 0, 'crosswalk': 1, 'sidewalk': 2,
    'stairs': 3, 'traffic_island': 4, 'zebra': 5
}

def generate_category_dict(categories):
    return {cat['name']: cat['id'] for cat in categories}

def invert_dict(d):
    return {v: k for k, v in d.items()}

def get_segmentation_mapping(category_dict):
    seg_map = {}
    for name, class_id in category_dict.items():
        if name in SEGMENTATION_CLASSES:
            seg_map[class_id] = SEGMENTATION_CLASSES[name]
    return seg_map

def convert(json_dirs, out_root, task_name='combined_task'):
    all_data = []
    skipped_files = []

    for jd in json_dirs:
        json_file = os.path.join(jd, 'result.json')
        if not os.path.exists(json_file):
            continue

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        images = {img['id']: img for img in data['images']}
        annotations = data['annotations']
        cat_dict = generate_category_dict(data['categories'])
        inv_cat_dict = invert_dict(cat_dict)
        seg_mapping = get_segmentation_mapping(cat_dict)

        image_anns = {img_id: [] for img_id in images}
        for ann in annotations:
            image_anns[ann['image_id']].append(ann)

        for img_id, anns in image_anns.items():
            img_info = images[img_id]
            file_base = os.path.splitext(img_info['file_name'])[0].split("\\")[-1]
            img_path = os.path.join(jd, 'images', file_base + '.jpg')

            if not os.path.exists(img_path):
                skipped_files.append(img_path)
                continue

            det_lines = []
            seg_lines = []

            for ann in anns:
                cat_id = ann['category_id']
                cat_name = inv_cat_dict.get(cat_id)

                if 'bbox' in ann and cat_name in DETECTION_CLASSES:
                    x, y, w, h = ann['bbox']
                    xc, yc = x + w/2, y + h/2
                    norm = [xc / img_info['width'], yc / img_info['height'], w / img_info['width'], h / img_info['height']]
                    det_lines.append(f"{DETECTION_CLASSES[cat_name]} " + " ".join(f"{v:.6f}" for v in norm))

                if 'segmentation' in ann and cat_id in seg_mapping:
                    for seg in ann['segmentation']:
                        norm_pts = [f"{seg[i]/img_info['width']:.6f} {seg[i+1]/img_info['height']:.6f}" for i in range(0, len(seg), 2)]
                        seg_lines.append(f"{seg_mapping[cat_id]} " + " ".join(norm_pts))

            if det_lines or seg_lines:
                all_data.append((img_path, det_lines, seg_lines))

    print("Total valid images:", len(all_data))

    # Split dataset
    random.shuffle(all_data)
    split = int((1 - val_ratio) * len(all_data))
    train_data = all_data[:split]
    val_data = all_data[split:]

    def save_dataset(dataset, split_name):
        for idx, (img_path, det_labels, seg_labels) in enumerate(dataset):
            img_out = os.path.join(out_root, 'images', split_name, f"{idx:07d}.jpg")
            shutil.copy(img_path, img_out)

            if det_labels:
                with open(os.path.join(out_root, 'detection', 'labels', split_name, f"{idx:07d}.txt"), 'w') as f:
                    f.write("\n".join(det_labels))
            if seg_labels:
                with open(os.path.join(out_root, 'segmentation', 'labels', split_name, f"{idx:07d}.txt"), 'w') as f:
                    f.write("\n".join(seg_labels))

    # Create directories
    for sub in ['images/train', 'images/val', 'detection/labels/train', 'detection/labels/val',
                'segmentation/labels/train', 'segmentation/labels/val']:
        os.makedirs(os.path.join(out_root, sub), exist_ok=True)

    save_dataset(train_data, 'train')
    save_dataset(val_data, 'val')

    print(f"Train: {len(train_data)} | Val: {len(val_data)}")
    print("Skipped images:", len(skipped_files))

# Example usage
input_path = "C:/Users/Administrator/Documents/VASCO/unzip"
json_dirs = [os.path.join(input_path, d) for d in os.listdir(input_path)]
output_root = "C:/Users/Administrator/Documents/VASCO/processed_multi"
convert(json_dirs, output_root)
