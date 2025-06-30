import os
import sys
sys.path.append("/home/nafisur/Documents/VASCO/custom_yolo/ultralytics")  # Adjust path to import from parent directory
import argparse
import yaml
from ultralytics.data.dataset import YOLODataset  # Adjust path to import from parent directory
from loss import v8DetectionLoss, v8SegmentationLoss
from model import MultiTaskYOLO  # Adjust path to import from parent directory
import torch
from tqdm import tqdm

img_path = '/home/nafisur/Documents/VASCO/processed_multi/images/train'
batch = 64
mode = "train"
# cfg = "config"
stride = 32

device = "cuda" if torch.cuda.is_available() else "cpu"
# read params from a JSON file
params_file = '/home/nafisur/Documents/VASCO/custom_yolo/cfg.yaml'
with open(params_file) as f:
    params = yaml.safe_load(f)

params = argparse.Namespace(**params)  # Convert dict to Namespace for compatibility with YOLODataset
# print(params)

model_params_file = '/home/nafisur/Documents/VASCO/custom_yolo/model_det_hyp.yaml'
with open(model_params_file) as f:
    model_params = yaml.safe_load(f)
    
model_params = argparse.Namespace(**model_params)  # Convert dict to Namespace for compatibility with YOLODataset
# print(model_params.box)
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
# initialize the model
model = MultiTaskYOLO().to(device)
model.det_head_layers.dfl.conv.weight.requires_grad_(True)
# initialize the loss function
loss_fn_det = v8DetectionLoss(model, model_params, device=device)
loss_fn_seg = v8SegmentationLoss(model, model_params, device=device)
# test_dataloader = dataset.test_dataloader()

from torch.utils.data import DataLoader
test_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=dataset.collate_fn)

# iterator = iter(test_dataloader)
# batch = next(iterator)

# print("done")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
accumulation_steps = 64  # Adjust as needed
epoch = 30
print(len(test_dataloader))
model.train()  # Set the model to training mode

for epoch_idx in range(epoch):
    running_loss = 0.0
    epoch_loss = 0.0
    total_samples = 0
    
    pbar = tqdm(test_dataloader, desc=f"Epoch {epoch_idx + 1}/{epoch}", leave=True, unit="batch")
    for batch_idx,batch in enumerate(pbar):
        det = batch['det']
        seg = batch['seg']
        img = batch['det']['img'].to(device).float() / 255
        
        output_dict = model(img)
        det_out = output_dict['det']
        seg_out = output_dict['seg']
        
        det_loss, det_loss_items = loss_fn_det(det_out, det)
        seg_loss, seg_loss_items = loss_fn_seg(seg_out, seg)
        
        # optimizer.zero_grad()
        batch_loss = (det_loss.sum() + seg_loss.sum()) / accumulation_steps  # Combine detection and segmentation losses
        # batch_loss = total_loss.sum() 
        batch_loss.backward()
        # if( batch_idx + 1 )% accumulation_steps == 0 or batch_idx + 1 == len(test_dataloader):
        #     optimizer.step()
        #     optimizer.zero_grad()
            # scheduler.step()  # Update learning rate
            
        optimizer.step()
        optimizer.zero_grad()
        # scheduler.step()  # Update learning rate scheduler
        # for name,param in model.named_parameters():
        #    if param.grad is None:
        #        print(f'{name} does not❌ have grad')
        #import pdb; pdb.set_trace()  # Debugging point to inspect gradients
        batch_size = img.size(0)
        running_loss += batch_loss.item() * batch_size
        total_samples += batch_size
        # print the det and seg losS
        pbar.set_postfix({
            "running Loss": f"{running_loss / total_samples:.4f}",
            "Epoch": f"{epoch_idx + 1}/{epoch}",
            "Batch": f"{batch_idx + 1}/{len(test_dataloader)}",
            "det_loss": f"{det_loss.sum().item() / batch_size:.4f}",
            "seg_loss": f"{seg_loss.sum().item() / batch_size:.4f}",
        })
    epoch_loss = running_loss/total_samples
    print(f"\nEpoch {epoch_idx + 1} completed. Average Loss: {epoch_loss:.4f}")  # Newline after epoch
    scheduler.step()  # Update learning rate scheduler
# save the model after training
torch.save(model.state_dict(), '/home/nafisur/Documents/VASCO/custom_yolo/multi_yolo_100.pth')
print("Training completed.")
