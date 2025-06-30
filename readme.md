### path
/Documents/VASCO/custom_yolo : for multi task yolo.
/Documents/VASCO/det_seg_yolo : for seperate yolo model with trained weights.

<pre>
import sys
sys.path.append("/home/nafisur/Documents/VASCO/custom_yolo/ultralytics")</pre>

add the path to the ultralytic file in files to use the changed files for multi task yolo.

- convert_coco_yolo.py file contains script to prepare the data set.

### Dataloader

to use the updated dataloader import it from 
`from ultralytics.data.dataset import YOLODataset`
(check train.py for reference) to make the dataloader work model params are required which is found in  '/home/nafisur/Documents/VASCO/custom_yolo/model_det_hyp.yaml' the dataloader return a dictonary `{det: {...}, seg: {...} }` where `det` and `seg` each have there own keys. Note the img in both the keys are the same use any for input into the model.
`det` has keys `dict_keys(['batch_idx', 'bboxes', 'cls', 'im_file', 'img', 'ori_shape', 'ratio_pad', 'resized_shape'])`
`seg` has keys `dict_keys(['batch_idx', 'bboxes', 'cls', 'im_file', 'img', 'masks', 'ori_shape', 'ratio_pad', 'resized_shape'])`
the collate_fn is updated to work with the new format.
### Note:

NoneScanning /home/nafisur/Documents/VASCO/processed_multi/detection/labels/train... 1848 images, 485 backgrounds, 0 corrupt: 100%|██████████| 2333/2333 [00:01<00:00, 1786
NoneNew cache created: /home/nafisur/Documents/VASCO/processed_multi/detection/labels/train.cache
NoneScanning /home/nafisur/Documents/VASCO/processed_multi/segmentation/labels/train... 2333 images, 0 backgrounds, 0 corrupt: 100%|██████████| 2333/2333 [00:01<00:00, 144
NoneNew cache created: /home/nafisur/Documents/VASCO/processed_multi/segmentation/labels/train.cache
**when running the code if there are corrupt files remove them from the image file so the dataloader is gives the correct labels**

### Model

the model is in the folder `model.py`
`from model import MultiTaskYOLO`
this gives the model with both segmentation and detection head.
In the file model.py file in `def __init__():` the model is created using yamal files from the `/home/nafisur/Documents/VASCO/custom_yolo/ultralytics/ultralytics/cfg/models/12` change the number of classes in the the yaml files to change the number of classes to predict (according to the dataset).

### Transforms

To make use of all the transforms available the file need to be updated according to the new format of the dataloader.

### loss

The loss file in the main directory (not in ultralytics folder) contains the changed codes. search # nafis new code for where changes are.

search `# nafis new code` for all the places changes are made.
the detection results are visualized the segmeation results needs to be visualized to validate if the seg head is learning.

### training
train.py has code for training. The segmentation head still needs work. The model doesn't seems to learn the segmentation task properly.