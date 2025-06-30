import os
import sys
sys.path.append("/home/nafisur/Documents/VASCO/custom_yolo/ultralytics")  # Adjust path to import from parent directory
import argparse
import yaml
from ultralytics.data.dataset import YOLODataset  # Adjust path to import from parent directory
from loss import v8DetectionLoss, v8SegmentationLoss
from torch.utils.data import DataLoader
from model import MultiTaskYOLO  # Adjust path to import from parent directory
import torch
from tqdm import tqdm

img_path = '/home/nafisur/Documents/VASCO/processed_multi/images/train'
val_img_path = '/home/nafisur/Documents/VASCO/processed_multi/images/val'
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

dataset_val = YOLODataset(img_path=val_img_path,
        imgsz=640,
        batch_size=batch,
        augment=mode == "test",  # augmentation   
        hyp=params,  # TODO: probably add a get_hyps_from_cfg function
        rect=False,  # rectangular batches
        cache=False,
        single_cls= False,
        stride=int(stride),
        pad=0.0,
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

test_dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=dataset.collate_fn)

val_dataloader = DataLoader(dataset_val, batch_size=32, shuffle=False, num_workers=4, collate_fn=dataset_val.collate_fn)
# iterator = iter(test_dataloader)
# batch = next(iterator)

# print("done")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
accumulation_steps = 64  # Adjust as needed
epoch = 100
print(len(test_dataloader))
model.train()  # Set the model to training mode
best_val_loss = float('inf')  # Initialize best validation loss

for epoch_idx in range(epoch):
    running_loss = 0.0
    epoch_loss = 0.0
    total_samples = 0
    
    pbar = tqdm(test_dataloader, desc=f"Epoch {epoch_idx + 1}/{epoch}", leave=True, unit="batch")
    for batch_idx,batch in enumerate(pbar):
        model.train()  # Ensure model is in training mode
        det = batch['det']
        seg = batch['seg']
        img = batch['det']['img'].to(device).float() / 255
        
        output_dict = model(img)
        det_out = output_dict['det']
        seg_out = output_dict['seg']
        
        det_loss, det_loss_items = loss_fn_det(det_out, det)
        seg_loss, seg_loss_items = loss_fn_seg(seg_out, seg)
        
        batch_loss = (det_loss.sum() + seg_loss.sum()) # Combine detection and segmentation losses
        # batch_loss = total_loss.sum() 
        batch_loss.backward()
       
            
        optimizer.step()
        optimizer.zero_grad()
        
        batch_size = img.size(0)
        running_loss += batch_loss.item()
        total_samples += batch_size
        
        current_lr = scheduler.get_last_lr()[0]
        
        pbar.set_postfix({
            "running Loss": f"{running_loss / total_samples:.4f}",
            "Epoch": f"{epoch_idx + 1}/{epoch}",
            "Batch": f"{batch_idx + 1}/{len(test_dataloader)}"
        })
    epoch_loss = running_loss/total_samples
    print(f"\nEpoch {epoch_idx + 1} completed. Average Loss: {epoch_loss:.4f}") # Newline after epoch
    print(f"Current Learning Rate: {current_lr:.6f}")
    scheduler.step()  # Update learning rate scheduler after each epoch

    model.eval()  # Set the model to evaluation mode
    val_running_loss = 0.0
    val_total_samples = 0
    with torch.no_grad():
        for val_batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch_idx + 1}", leave=False, unit="batch"):
            val_det = val_batch['det']
            val_seg = val_batch['seg']
            val_img = val_batch['det']['img'].to(device).float() / 255
            
            val_output_dict = model(val_img)
            val_det_out = val_output_dict['det']
            val_seg_out = val_output_dict['seg']
            
            val_det_loss, _ = loss_fn_det(val_det_out, val_det)
            val_seg_loss, _ = loss_fn_seg(val_seg_out, val_seg)
            
            val_batch_loss = (val_det_loss.sum() + val_seg_loss.sum())
            batch_size = val_img.size(0)
            val_running_loss += val_batch_loss.item() 
            val_total_samples += batch_size
        val_epoch_loss = val_running_loss / val_total_samples
        print(f"Validation Loss: {val_epoch_loss:.4f}")
    # Save the model if validation loss improves
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        print(f"Validation loss improved to {best_val_loss:.4f}. Saving model...")
        os.makedirs('runs2', exist_ok=True)
        torch.save(model.state_dict(), 'runs2/best_model.pth')  

# save the model after training
os.makedirs('runs2', exist_ok=True)
torch.save(model.state_dict(), 'runs2/batch_32_epoch_100.pth')
print("Training completed.")
