# Engilish
*  **Theory** : [https://wikidocs.net/216394](https://wikidocs.net/216394) <br>
*  **Implementation** : [https://wikidocs.net/227355](https://wikidocs.net/227355)

# 한글
*  **Theory** : [https://wikidocs.net/215124](https://wikidocs.net/215124) <br>
*  **Implementation** : [https://wikidocs.net/225901](https://wikidocs.net/225901)

This repository is folked from [https://github.com/Okery/PyTorch-Simple-MaskRCNN](https://github.com/Okery/PyTorch-Simple-MaskRCNN).
At this repository, simplification and explanation and will be tested at Colab Environment. Some bugs were fixed.


## Step 1. Clone from Github and install library

Git clone to root directory. 

```Shell
# Clone from Github Repository
! git init .
! git remote add origin https://github.com/RichardMinsooGo-ML/Bible_4_Part_K_05_Masked_R_CNN.git
# ! git remote add origin https://github.com/Okery/PyTorch-Simple-MaskRCNN.git
# ! git pull origin master
! git pull origin main
```

A tool to count the FLOPs of PyTorch model.

```
from IPython.display import clear_output
clear_output()
```

# 1. VOC dataset
## Download VOC Dataset

```Shell
# VOC 2012 Dataset Download and extract

! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
!tar -xvf "/content/VOCtrainval_11-May-2012.tar" -C "/content/dataset"
clear_output()

"""
! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

!tar -xvf "/content/VOCtrainval_06-Nov-2007.tar" -C "/content/dataset"
!tar -xvf "/content/VOCtest_06-Nov-2007.tar" -C "/content/dataset"
clear_output()
"""
```

## Filename change : It need to be fixed.

```
# This code is not generalized yet. So, it need to rename the files
import os

old_name = r"/content/dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"
new_name = r"/content/dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/train2017.txt"
os.rename(old_name, new_name)

old_name = r"/content/dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"
new_name = r"/content/dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/val2017.txt"
os.rename(old_name, new_name)
```

## Train

```
! python train.py --use-cuda --epochs 50 --iters 200 --dataset voc --data-dir /content/dataset/VOCdevkit/VOC2012
```

## Demo - Image

#### Filename change : It need to be fixed.

```
# This code is not generalized yet. So, it need to rename the files
old_name = r"/content/dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/val2017.txt"
new_name = r"/content/dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"
os.rename(old_name, new_name)

! mkdir image
```

#### Demo

```
import torch
import pytorch_mask_rcnn as pmr


use_cuda = True
dataset = "voc"
ckpt_path = "/content/maskrcnn_voc-50.pth"
data_dir  = "/content/dataset/VOCdevkit/VOC2012"

device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
if device.type == "cuda":
    pmr.get_gpu_prop(show=True)
print("\ndevice: {}".format(device))

ds = pmr.datasets(dataset, data_dir, "val", train=True)
#indices = torch.randperm(len(ds)).tolist()
#d = torch.utils.data.Subset(ds, indices)
d = torch.utils.data.DataLoader(ds, shuffle=False)

model = pmr.maskrcnn_resnet50(True, max(ds.classes) + 1).to(device)
model.eval()
model.head.score_thresh = 0.3

if ckpt_path:
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(checkpoint["eval_info"])
    del checkpoint

for p in model.parameters():
    p.requires_grad_(False)

iters = 3

for i, (image, target) in enumerate(d):
    image = image.to(device)[0]
    #target = {k: v.to(device) for k, v in target.items()}

    with torch.no_grad():
        result = model(image)

    pmr.show(image, result, ds.classes, "./image/output{}.jpg".format(i))

    if i >= iters - 1:
        break
```

## Evaluation - voc

```
! python eval.py --ckpt-path /content/maskrcnn_voc-50.pth \
                 --dataset voc \
                 --data-dir /content/dataset/VOCdevkit/VOC2012
```

# 2. COCO dataset
## Download COCO Dataset

```

! wget http://images.cocodataset.org/zips/train2017.zip
! wget http://images.cocodataset.org/zips/val2017.zip
# ! wget http://images.cocodataset.org/zips/test2017.zip
# ! wget http://images.cocodataset.org/zips/unlabeled2017.zip


! unzip train2017.zip  -d dataset/COCO2017
! unzip val2017.zip  -d dataset/COCO2017
clear_output()

# ! unzip test2017.zip
# clear_output()

# ! unzip unlabeled2017.zip
# clear_output()

# ! rm train2017.zip
# ! rm val2017.zip
# ! rm test2017.zip
# ! rm unlabeled2017.zip

! wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
# wget http://images.cocodataset.org/annotations/image_info_test2017.zip
# wget http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip

! unzip annotations_trainval2017.zip -d dataset/COCO2017
# ! unzip stuff_annotations_trainval2017.zip
# ! unzip image_info_test2017.zip
# ! unzip image_info_unlabeled2017.zip

# ! rm annotations_trainval2017.zip
# ! rm stuff_annotations_trainval2017.zip
# ! rm image_info_test2017.zip
# ! rm image_info_unlabeled2017.zip
```

## Train

```
! python train.py --use-cuda --epochs 50 --iters 200 --dataset coco --data-dir dataset/COCO2017
```

## Demo - Image
#### Filename change : It need to be fixed.

```
# This code is not generalized yet. So, it need to rename the files
old_name = r"/content/dataset/COCO2017/annotations/instances_val2017.json"
new_name = r"/content/dataset/COCO2017/annotations/instances_val.json"
os.rename(old_name, new_name)

# This code is not generalized yet. So, it need to rename the files
old_name = r"/content/dataset/COCO2017/val2017"
new_name = r"/content/dataset/COCO2017/val"
os.rename(old_name, new_name)
```

#### Demo-Images
```
import torch
import pytorch_mask_rcnn as pmr

use_cuda = True
dataset = "coco"
ckpt_path = "/content/maskrcnn_coco-50.pth"
data_dir  = "/content/dataset/COCO2017"

device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
if device.type == "cuda":
    pmr.get_gpu_prop(show=True)
print("\ndevice: {}".format(device))

ds = pmr.datasets(dataset, data_dir, "val", train=True)
#indices = torch.randperm(len(ds)).tolist()
#d = torch.utils.data.Subset(ds, indices)
d = torch.utils.data.DataLoader(ds, shuffle=False)

model = pmr.maskrcnn_resnet50(True, max(ds.classes) + 1).to(device)
model.eval()
model.head.score_thresh = 0.3

if ckpt_path:
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(checkpoint["eval_info"])
    del checkpoint

for p in model.parameters():
    p.requires_grad_(False)

iters = 3

for i, (image, target) in enumerate(d):
    image = image.to(device)[0]
    #target = {k: v.to(device) for k, v in target.items()}

    with torch.no_grad():
        result = model(image)

    pmr.show(image, result, ds.classes, "./image/output{}.jpg".format(i))

    if i >= iters - 1:
        break
```

## Evaluation - coco

```
ckpt_path = "/content/maskrcnn_coco-50.pth"
data_dir  = "/content/dataset/COCO2017"
! python eval.py --ckpt-path ckpt_path \
                 --dataset coco \
                 --data-dir data_dir
```


