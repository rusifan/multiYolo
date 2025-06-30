import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/nafisur/Documents/VASCO/custom_yolo/ultralytics")  # Adjust path to import from parent directory
from model import MultiTaskYOLO
from ultralytics import YOLO
import torch
import argparse
import yaml
from ultralytics.data.dataset import YOLODataset  # Adjust path to import from parent directory
from ultralytics.engine.validator import BaseValidator
from ultralytics.engine.predictor import BasePredictor
from ultralytics.data.build import load_inference_source
img_path = '/home/nafisur/Documents/VASCO/processed_multi/images/val'
batch = 64
mode = "val"
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
model = MultiTaskYOLO().to(device)
# model = YOLO("yolo12n.yaml")  # YAML defines model architecture
# model = model.model.cuda() 
# Load the model weights
model_weight_path = '/home/nafisur/Documents/VASCO/custom_yolo/mullti_weight/batch_32_epoch_100.pth'
# model_weight_path = '/home/nafisur/Documents/VASCO/custom_yolo/runs2/batch_16_epoch_150.pth'
# model_weight_path = '/home/nafisur/Documents/VASCO/custom_yolo/best_model.pth'
if os.path.exists(model_weight_path):
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
else:
    raise FileNotFoundError(f"Model weights not found at {model_weight_path}")

preditor = BasePredictor()

# val_img = dataset[0]['img']
val_img_path= "/home/nafisur/Documents/VASCO/processed_multi/images/val/0000035.jpg"
# org_img = val_img.unsqueeze(0).to(device)  # Add batch dimension and move to device
dataset = load_inference_source(val_img_path)
# load the validation image from path
val_img = cv2.imread(val_img_path)
height, width = val_img.shape[:2]
val_img_copy = cv2.imread(val_img_path)
# convert the image to tensor and normalize it
org_img = torch.from_numpy(val_img_copy).float().to(device).float() / 255.0  # Normalize the image

val_img = cv2.resize(val_img, (640, 640))  # Resize to model input size
val_img = torch.from_numpy(val_img).float().to(device).float() / 255.0  # Normalize the image
val_img = val_img.permute(2, 0, 1).unsqueeze(0)  # Change to (N, C, H, W) format

for batch in dataset:
    paths, im0s, s = batch
    # im is a tensor of im0s / 255
    # create im from im0 that is resized to the model input size, normalized, and transposed 
    # im = torch.from_numpy(im0s).float().to(device).float() / 255.0  # Normalize the image
    # val_img = im # Normalize the image
print(f"Image shape:{val_img_copy.shape}")
# Perform inference
model.eval()
with torch.no_grad():
    output = model(val_img)
# Post-process predictions
# print(f"Predictions: {preds}")
seg_preds = output['seg']  # Assuming the model returns a list of segmentation predictions
# print(f"Segmentation Predictions: {seg_preds}")
det_preds = output['det']  # Assuming the model returns a list of predictions
# det_preds = output  # Assuming the model returns a list of predictions
from ultralytics.utils import ops
preds = ops.non_max_suppression(
    det_preds,
    conf_thres=0.15,
    iou_thres=0.5,
    classes=None,
    agnostic=False,
    max_det=1000,
    nc=14,  # Number of classes
    end2end=False,
    rotated=False,
)
print(f"Detection Predictions: {preds}")
# scale the bounding boxes to the original image size
bboxes = [pred[:, :4] for pred in preds]  # Extract bounding boxes from predictions
res = []
for i, bbox in enumerate(bboxes):
    print(f"Bounding boxes for image {i}: {bbox}")
    x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    # rescale the coordinated divide by 640 and scale it to height and width
    x1 = x1 * width / 640
    y1 = y1 * height / 640
    x2 = x2 * width / 640
    y2 = y2 * height / 640
    res.append(torch.stack((x1, y1, x2, y2), dim=1))  # Stack the coordinates into a tensor
    

img = cv2.imread(val_img_path)  # Load the original image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for plotting
# img = cv2.resize(img, (640, 640))  # Resize to match the model input size
plt.figure(figsize=(10, 10))
res = np.array(res[0].cpu())  # Convert the first result to numpy for plotting
for box in res:
    x1, y1, x2, y2 = box
    print(f"Bounding box coordinates: {x1}, {y1}, {x2}, {y2}")
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Draw the bounding box on the image
# plt.imshow(img)
cv2.imwrite('output32.png', img)  # Save the plot as an image
plt.axis('off')
pred_seg = ops.non_max_suppression(
    seg_preds,
    conf_thres=0.25,
    iou_thres=0.75,
    classes=None,
    agnostic=False,
    max_det=300,
    nc=6,  # Number of classes
    end2end=False,
    rotated=False,
)
# print(f"Segmentation Predictions: {pred_seg}")
# Assuming pred_seg is a list of segmentation predictions, we can visualize them
img = cv2.imread(val_img_path)  # Load the original image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert B
for i, seg in enumerate(pred_seg):
    if seg is not None and len(seg) > 0:
        # Convert segmentation masks to numpy array
        seg_mask = seg[0].cpu().numpy()  # Assuming the first element is the mask
        seg_mask = (seg_mask * 255).astype(np.uint8)  # Scale to 0-255 for visualization
        seg_mask_resize = cv2.resize(seg_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)  # Resize to match the original image size
        # draw the contours of the segmentation mask
        contours, _ = cv2.findContours(seg_mask_resize, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) > 0:
                cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
        break
        
        # break
# Save the final image with bounding boxes and segmentation masks
cv2.imwrite('output_with_segmentation.png', img)  # Save the final image with
print('Done')

from ultralytics.engine.results import Results
def plot_results(results, img_path):
    """
    Plot the results on the original image.

    Args:
        results (Results): Results object containing the predictions.
        img (np.ndarray): Original image before preprocessing.
    """
    img = cv2.imread(img_path)  # Load the original image
    width, height = img.shape[1], img.shape[0]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for plotting
    plt.figure(figsize=(10, 10))
    # plt.imshow(img)
    ax = plt.gca()
    results = results.cpu().numpy()  # Ensure results are on CPU for plotting    
    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = box[:4]
        print(f"Bounding box coordinates: {x1}, {y1}, {x2}, {y2}")
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2) 
        # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Draw the bounding box on the image     
        # rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)
    
    plt.axis('off')
    cv2.imwrite('output_result.png', img)  # Save the plot as an image
    # plt.savefig('results_plot.png')  # Save the plot as an image
#     # plt.show()
def construct_result(pred, img, orig_img, img_path):
        """
        Construct a single Results object from one image prediction.

        Args:
            pred (torch.Tensor): Predicted boxes and scores with shape (N, 6) where N is the number of detections.
            img (torch.Tensor): Preprocessed image tensor used for inference.
            orig_img (np.ndarray): Original image before preprocessing.
            img_path (str): Path to the original image file.

        Returns:
            (Results): Results object containing the original image, image path, class names, and scaled bounding boxes.
        """
        # img = img[0]
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], [1080, 1920])
        return Results(orig_img, path=img_path, names="multi", boxes=pred[:, :6])

reuslt = [ construct_result(pred, val_img, orig_img, img_path) 
          for pred, orig_img, img_path in zip(preds, org_img, batch) ]
print(reuslt[0].boxes.xyxy)  # Print the bounding boxes of the first result
# plot the results on the original image
# print(f'bbox {reuslt[0]}')

res = reuslt[0].boxes.xyxy.cpu().numpy()  # Convert to numpy for plotting
plot_results(reuslt[0], val_img_path)  # Plot the results on the original image