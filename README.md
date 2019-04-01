# FPN + OHEM

It is a code base for single stage Feature Pyramid Network (FPN) with online hard example mining (OHEM). 

It is a pure [Pytorch 1.0](https://pytorch.org/) code, including preprocessing.

## Introduction 

This repository contains a single stage version of FPN present on [RetinaNet paper](https://arxiv.org/pdf/1708.02002.pdf).
Objective to reproduce Table 1 with ResNet50 with OHEM.

## Archtecture 

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

Dataset | Backbone | Type | #Anchors | Batch-Size | lr | #iter | #GPUs| mAP@0.5 % | 
|-------| :----: | :----: | :-----:  |  :----: | :----: | :-----:  | :---:|
| VOC | ResNet50 | Pre-defined | 9 | 16 | 0.0002 | 70K  | 2 | 78.1 |
| VOC | ResNet50 | Pre-defined | 3 | 16 | 0.0002 |  70K |  2 | 77.2 |
| VOC | ResNet50 | Clustered | 3 | 16 | 0.0002 |  70K |  2 | 79.5 |
| COCO | ResNet50 | Pre-defined | 9 | 32 | 0.0005 |  -- |  4 | training |
| COCO | ResNet50 | Pre-defined | 3 | 32 | 0.0005 | -- | 4 | up-next |
| COCO | ResNet50 | Clustered | 3 | 32 | 0.0005 | -- |  4 | up-next |

#### Results of COCO are coming soon!
 
## Details
- Max-pooling in resnet after first convolutional layer is not used
- Input image size is `300`.
- Resulting feature map size on five pyramid levels is `[75, 38, 19, 10, 5]` 
- Learning rate dropped after 50K iterations in case of VOC
- Learning rate dropped after 120K iterations in case of COCO



## Installation
- We used anaconda 3.7 as python distribution
- You will need [Pytorch1.0](https://pytorch.org/get-started/locally/)
- visdom and tensorboardX if you want to use the visualisation of loss and evaluation
  -- if do want to use them set visdom/tensorboard flag equal to true while training 
  -- and configure the visdom port in arguments in  `train.py.`
- OpenCV is needed as well, install it using `conda install opencv.`

## TRAINING
Please follow dataset preparation [README](https://github.com/gurkirt/FPN.pytorch/tree/master/prep) from `prep` folder of this repo.
Once you have pre-processed the dataset, then you are ready to train your networks.

To train run the following command. 

`python train.py --dataset=voc --basenet=resnet50 --batch_size=16 --lr=0.0002 -j=8  --ngpu=2 --step_values=40000 --max_iter=50000`

It will use all the visible GPUs. You can append `CUDA_VISIBLE_DEVICES=gpuids-comma-separated` at the beginning of the above command to mask certain GPUs. We used two GPU machine to run these experiments.

Please check the arguments in `train.py` to adjust the training process to your liking.

## Evaluation
Model is evaluated and saved after each `10K` iterations. 

mAP@0.5 is computed after every 10K iterations and at the end.

Demo script and coco evaluation protocol will be updated in the coming weeks.

## Training on a custom dataset.
You can take inspiration from data preparation scripts from `prep` directory, which we used to pre-process VOC and COCO dataset.

Also checkout [README](https://github.com/gurkirt/FPN.pytorch/tree/master/prep) in `prep` directory.

If you want to use clustered anchors, then you can use from either of existing anchors or cluster the anchors yourself using `kmeans_for_anchors.py`.

## References
[RetinaNet paper](https://arxiv.org/pdf/1708.02002.pdf)
[Our realtime-action-detection (ROAD) system implemetation](https://github.com/gurkirt/realtime-action-detection)