import sys
sys.path.append("/home/nafisur/Documents/VASCO/custom_yolo/ultralytics")  # Adjust path to import from parent directory


from ultralytics import YOLO
import torch
from torch import nn
from ultralytics.nn.modules.conv import Concat  # Required to handle skip-connections

from torch.utils.data import DataLoader
from torchvision import transforms

import os
import torch
from torch.utils.data import Dataset
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model_det = YOLO("yolo12n.yaml").to(device)  # Load a pretrained YOLOv8n model
model_seg = YOLO("yolo12n-seg.yaml").to(device)

class MultiTaskYOLO(nn.Module):
    def __init__(self, model_seg, model_det):
        super().__init__()
        self.backbone_layers = list(model_seg.model.model[:9])   # Backbone
        self.seg_head_layers = list(model_seg.model.model[9:])   # Segmentation head
        self.det_head_layers = model_det.model.model[-1]   # Detection head

    def forward(self, x):
        intermediates = []
        out = x
        for layer in self.backbone_layers:
            out = layer(out)
            intermediates.append(out)

        head_outs = []
        x = self.seg_head_layers[0](intermediates[8])            # Upsample
        x = self.seg_head_layers[1]([x, intermediates[6]])       # Concat P4
        x = self.seg_head_layers[2](x)                           # 11
        head_outs.append(x)  # x11

        x = self.seg_head_layers[3](x)                           # Upsample
        x = self.seg_head_layers[4]([x, intermediates[4]])       # Concat P3
        x = self.seg_head_layers[5](x)                           # 14
        head_outs.append(x)  # x14

        x = self.seg_head_layers[6](x)                           # Downsample
        x = self.seg_head_layers[7]([x, head_outs[0]])           # Concat with x11
        x = self.seg_head_layers[8](x)                           # 17
        head_outs.append(x)  # x17

        x = self.seg_head_layers[9](x)                           # Downsample
        x = self.seg_head_layers[10]([x, intermediates[8]])      # Concat P5
        x = self.seg_head_layers[11](x)                          # 20
        head_outs.append(x)  # x20

        det = self.det_head_layers([head_outs[1], head_outs[2], head_outs[3]])  # Detect([14,17,20])
        seg = self.seg_head_layers[12]([head_outs[1], head_outs[2], head_outs[3]])  # Segment([14,17,20])
        return {"seg": seg, "det": det}

if __name__ == "__main__":
    model = MultiTaskYOLO(model_seg, model_det).to(device)
    model.eval()

    # Example input
    dummy_input = torch.randn(1, 3, 640, 640).to(device)

    with torch.no_grad():
        output = model(dummy_input)
        print("Detection Output:", len(output["det"]))
        print("Segmentation Output:", len(output["seg"]))