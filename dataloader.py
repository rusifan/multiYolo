import os
import glob
from pathlib import Path
from typing import List
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

from utils import multi_img2label_paths,IMG_FORMATS

class MultiTaskDataset(Dataset):
    def __init__(self, im_files, det_label_dir, seg_label_dir, img_size):
        self.im_files = im_files
        self.det_label_dir = det_label_dir
        self.seg_label_dir = seg_label_dir
        self.img_size = img_size

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomRotation(degrees=15),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.RandomGrayscale(p=0.1),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.im_files)

    def load_txt_labels(self, label_path):
        if not os.path.exists(label_path):
            return []

        with open(label_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        return [[float(x) for x in line.split()] for line in lines]

    def __getitem__(self, idx):
        # Load and transform image
        img_path = self.im_files[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Load detection labels
        det_anns = self.load_txt_labels(self.det_label_dir[idx])
        det_tensor = torch.tensor(det_anns, dtype=torch.float32) if det_anns else torch.empty((0, 5))

        # Load segmentation labels
        seg_anns = self.load_txt_labels(self.seg_label_dir[idx])
        seg_tensor = [torch.tensor(ann, dtype=torch.float32) for ann in seg_anns] if seg_anns else []

        return image, det_tensor, seg_tensor

def custom_collate_fn(batch):
    images, detections, segmentations = zip(*batch)
    images = torch.stack(images)
    return images, detections, segmentations


def get_multi_labels(im_files):
    """Returns dictionary of labels for YOLO training."""
    # label_list = []
    label_files = []
    for task_name in ['detection','segmentation']:
        label_files_names = multi_img2label_paths(im_files,task_name)
        label_files.append(label_files_names)
    return label_files

# def v8_transforms(dataset, imgsz, hyp):
#     """Convert images to a size suitable for YOLOv8 training."""
#     pre_transform = Compose([
#         Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic),
#         CopyPaste(p=hyp.copy_paste),
#         RandomPerspective(
#             degrees=hyp.degrees,
#             translate=hyp.translate,
#             scale=hyp.scale,
#             shear=hyp.shear,
#             perspective=hyp.perspective,
#             pre_transform=LetterBox(new_shape=(imgsz, imgsz)),
#         )])
#     flip_idx = dataset.data.get('flip_idx', None)  # for keypoints augmentation
#     if dataset.use_keypoints and flip_idx is None and hyp.fliplr > 0.0:
#         hyp.fliplr = 0.0
#         LOGGER.warning("WARNING ⚠️ No `flip_idx` provided while training keypoints, setting augmentation 'fliplr=0.0'")
#     return Compose([
#         pre_transform,
#         MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
#         Albumentations(p=1.0),
#         RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
#         RandomFlip(direction='vertical', p=hyp.flipud),
#         RandomFlip(direction='horizontal', p=hyp.fliplr, flip_idx=flip_idx)])  # transforms


# def build_transforms(self, hyp=None):
#         """Builds and appends transforms to the list."""
#         if self.augment:
#             hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
#             hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
#             transforms = v8_transforms(self, self.imgsz, hyp)
#         else:
#             transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
#         transforms.append(
#             Format(bbox_format='xywh',
#                    normalize=True,
#                    return_mask=self.use_segments,
#                    return_keypoint=self.use_keypoints,
#                    batch_idx=True,
#                    mask_ratio=hyp.mask_ratio,
#                    mask_overlap=hyp.overlap_mask,
#                    labels_name=self.data['labels_list']))
#         return transforms
def build_dataloader(im_files, labels, img_size, batch_size, num_workers):
    dataset = MultiTaskDataset(im_files, labels[0], labels[1], img_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
        )  
    return dataloader

def get_image_files(img_path):
    img_list = []
    for root, _, files in os.walk(img_path):
        for file in files:
            if file.lower().endswith(tuple(IMG_FORMATS)):
                img_list.append(os.path.join(root, file))
    return img_list
     
def build_dataset(img_path, img_size, batch_size, num_workes):
    im_files = get_image_files(img_path)
    labels = get_multi_labels(im_files)
    
    return im_files, labels


if __name__ == "__main__":
    # Example usage
    img_path = "/home/nafisur/Documents/VASCO/processed_multi/images"
    train_path = os.path.join(img_path, 'train')
    val_path = os.path.join(img_path, 'val')
    
    img_size = 640
    batch_size = 1
    num_workers = 4

    im_files, labels = build_dataset(train_path, img_size, batch_size, num_workers)
    train_dataloader = build_dataloader(im_files, labels, img_size, batch_size, num_workers)
    
    for images, det_labels, seg_labels in train_dataloader:
        print(f"Batch size: {len(images)}")
        print(f"Detection labels: {det_labels}")
        print(f"Segmentation labels: {seg_labels}")
        break

    