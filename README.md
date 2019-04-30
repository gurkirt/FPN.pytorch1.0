# FPN + OHEM 

It is a code base for single stage Feature Pyramid Network (FPN) with online hard example mining (OHEM). 
We implement shared heads, unlike in the paper. Shared heads help to reduce the memory consumption and improve the performance a little. 

It is a pure [Pytorch 1.0](https://pytorch.org/) code, including preprocessing of the input data. Annotations for both COCO and VOC dataset are provided in the same format. 

## Introduction 

This repository contains a single stage version of FPN present on [RetinaNet paper](https://arxiv.org/pdf/1708.02002.pdf).
Objective to reproduce Table 1 with ResNet50 with OHEM.

## Architecture 

![RetinaNet Structure](/figures/retinaNet.png)

ResNet is used as a backbone network (a) to build the pyramid features (b). 
Each classification (c) and regression (d) subnet is made of 4 convolutional layers and finally a convolutional layer to predict the class scores and bounding box coordinated respectively.

We freeze the batch normalisation layers of ResNet based backbone networks. 

## Loss function 
We use multi-box loss function with online hard example mining (OHEM), similar to [SSD](https://arxiv.org/pdf/1512.02325.pdf).
A huge thanks to Max DeGroot, Ellis Brown for [Pytorch implementation](https://github.com/amdegroot/ssd.pytorch) of SSD and loss function.


## Anchors
We use two types of anchor.
### Pre-defined anchors
Similar to RetinaNet, we can build anchor boxes or sometimes called prior boxes in SSD.
As a baseline, we use three aspect ratios (AR) and three scale ratios (SR) per pyramid-level, which results in nine anchors per cell location in the grid of each pyramid level.
It is resulting in `67K` total number of anchors/predictions. 

OR, we can have only one scale ratio. The reasoning behind that is the pyramid should be able to capture the scale space. 
Now, we will have three anchors per cell location in the grid of each pyramid level.
The total number of anchors/predictions close to `22K`. 
Although we will save the computational cost this way by predicting fewer boxes, the recall of result anchor boxes drops drastically.

### K-mean the anchors
An alternative is to boost the recall of one scale ratio by performing k-means on ground truth boxes with k = 3.
We pick three anchors from each pyramid level and initial cluster centre and then perform cluster on ground truth boxes.
Intersection-over-union (IoU) is used as a distance metric. 
Since the cluster centres and centred around the origin, we need to move the centre of each ground truth box to the origin as well.

We performed clustering for `coco` and `voc` independently.

### Average IoU and Recall
Here are the recall and average IoU obtained before and after clustering the anchors.

Dataset | Type | SR    | AR   | #Anchors/level | Total | Avergae IoU | Recall % |
|-------|:----: |:----:| :-----:  | :---:| :---:| :---:| :---: |
| VOC |  Pre-defined | 3    | 3   |  9     | 67K  |  0.78 | 96 |
| VOC |  Pre-defined | 2    | 3   |  6     | 44K  |  0.76 | 95 |
| VOC |  Pre-defined | 1    | 3   |  3     | 22K  |  0.66 | 88 |
| VOC |  Clustered   | 1    | 3   |  3     | 22K  |  0.74 | 97 |
| COCO |  Pre-defined | 3    | 3   |  9     | 67K  |  0.72 | 85 |
| COCO |  Pre-defined | 2    | 3   |  6     | 44K  |  0.69 | 85 |
| COCO |  Pre-defined | 1    | 3   |  3     | 22K  |  0.61 | 77 |
| COCO |  Clustered   | 1    | 3   |  3     | 22K  |  0.65 | 87 |


## Performance

There is a variation of the standard network where the features of localisation and classification heads are shared.

Dataset | Backbone | Type | #Anchors | mAP@0.5 % | Download |
|----|   :---: |     :---: | :---:  |  :---: | :---: | 
| VOC | ResNet50  | Pre-defined   | 9 |  81.3 | [link](https://drive.google.com/open?id=1elTmzdSTZOgY5_zJR-F2xl9_S9lbOn0A) |
| VOC | ResNet50  | Pre-defined   | 3 |  81.3  |  [link](https://drive.google.com/open?id=1VvneQZDxyw1ItbD1YbsWZ8kABzNy5hwc) |
| VOC | ResNet50  | Clustered     | 3 |  82.8 | [link](https://drive.google.com/open?id=10nbmVDA6UeC0H0GpIkEl3Uox4GWuEm6d) |
| VOC | ResNet50  | Clustered- SH | 3 |  82.7 | [link](https://drive.google.com/open?id=1mV62nGjtVq7ENTZ2-k9Y7hPD77scy5yw) |
| COCO | ResNet50 | Pre-defined   | 9 |  46.1 | [link](https://drive.google.com/open?id=1SYshc0QfGm9mV4SYaCdZIKpchLwW9sZx) |
| COCO | ResNet50 | Clustered     | 3 |  47.7 | [link](https://drive.google.com/open?id=17IaNr4xvhx9VBNuPhqGT85W-8IPyWam_) |
| COCO | ResNet50 | Clustered- SH | 3 |  48.3 | [link](https://drive.google.com/open?id=1yKQ7nxtaEsRjPAsGNfXswM365c8LGLzx) |

Here is [GoggleDrive](https://drive.google.com/open?id=1DmRjEUUqWgI2kTw3XM83J9Esbb7rO4dm) for all the above in signle folder.

Directory structure is similiar to one used in training setup. You can evaluate these models using `evaluate.py` and same hypermeter used in training, please read the arguments carefully.

## Details
- Input image size is `600`.
- Resulting feature map size on five pyramid levels is `[75, 38, 19, 10, 5]` 
- Batch size is set to `24`, the learning rate of `0.0005`.
- VOC, number of iterations are `50K`, and learning rate is dropped after `40K` iterations
- COCO, number of iterations are `150K`, and learning rate is dropped after `120K` iterations
- VOC can be trained in 2 TitanX GPUs, 12GB each
- COCO would need 3-4 GPUs because the number of classes is 80, hence loss function requires more memory
- `SH`, i.e. `Shared heads` helps to solve memory problem up to a point, but we will still need 2 GPUs to train on VOC or COCO

## Installation
- We used anaconda 3.7 as python distribution
- You will need [Pytorch1.0](https://pytorch.org/get-started/locally/)
- visdom and tensorboardX if you want to use the visualisation of loss and evaluation
  -- if you want to use them set visdom/tensorboard flag equal to true while training 
  -- and configure the visdom port in arguments in  `train.py.`
- OpenCV is needed as well, install it using `conda install opencv.`

## TRAINING
Please follow dataset preparation [README](https://github.com/gurkirt/FPN.pytorch/tree/master/prep) from `prep` folder of this repo.
Once you have pre-processed the dataset, then you are ready to train your networks.

To train run the following command. 

`python train.py --dataset=coco --basenet=resnet50 --batch_size=24 --lr=0.0005 -j=4  --ngpu=2 --step_values=120000 --max_iter=150000 --visdom=True --tensorboard=True --val_step=15000 --anchor_type=kmeans --shared_heads=1`

It will use all the visible GPUs. You can append `CUDA_VISIBLE_DEVICES=gpuids-comma-separated` at the beginning of the above command to mask certain GPUs. We used two GPU machine to run these experiments.

Please check the arguments in `train.py` to adjust the training process to your liking.

## Evaluation
Model is evaluated and saved after each `10K` iterations. 

mAP@0.5 is computed after every 10K iterations and at the end.

Coco evaluation protocol is demonstraed  in `evaluate.py` 

`python evaluate.py --dataset=coco --basenet=resnet50 --batch_size=24 --lr=0.0005 -j=2  --ngpu=2 --eval_iters=150000 --anchor_type=kmeans --shared_heads=1`

## COCO-API Result
Here are results COCO using [COCO-API](https://github.com/cocodataset/cocoapi) using final model with shared heads and kmeans based anchors.
Results using `cocoapi` are slightly different than above table.

```
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.285
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.492
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.293
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.138
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.318
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.391
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.258
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.412
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.436
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.251
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.479
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.575
```

## Training on a custom dataset.
You can take inspiration from data preparation scripts from `prep` directory, which we used to pre-process VOC and COCO dataset.

Also checkout [README](https://github.com/gurkirt/FPN.pytorch/tree/master/prep) in `prep` directory.

If you want to use clustered anchors, then you can either use existing anchors or cluster the anchors yourself using `kmeans_for_anchors.py`.

## References
- [RetinaNet paper](https://arxiv.org/pdf/1708.02002.pdf)
- [SSD paper for OHEM](https://arxiv.org/abs/1512.02325)
- [Our realtime-action-detection (ROAD) system implemetation](https://github.com/gurkirt/realtime-action-detection)