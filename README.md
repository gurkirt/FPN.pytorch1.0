# Single stage Feature Pyramid Network (FPN) with online hard example mining (OHEM)

This repository contains a single stage version of FPN present on [RetinaNet paper](https://arxiv.org/pdf/1708.02002.pdf).
Objective to reproduce Table 1 with ResNet50 with OHEM.

## Archtecture 

![RetinaNet Structure](/figures/retinaNet.png)

ResNet is used as a backbone network (a) to build the pyramid features (b). 
Each classification (c) and regression (d) subnet is made of 4 convolutional layers and finally a convolutional layer to predict the class scores and bounding box coordinated respectively.

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
Here is the recall and average IoU obtained before and after cluster anchors.

Dataset | SR    | AR   | #Anchors/level | Total | Isclustering | Avergae IoU % | Recall % |
|-------|:----: |:----:| :-----:           | :---:| :---:| :---:| :---: |
| COCO |  3    | 3   |  9     | 67K | [x] |  72 | 85 |

# It is getting ready, another couple of days: updated on 29th March

We freeze the batch normalisation layers of  

## Performance map@0.5 ResNet50

Kmean 3 anchors per scale: 79.7
 
## Results will be out soon on coco as well

## Installation
- We used anaconda 3.7 as python distribution
- You will need [Pytorch1.0](https://pytorch.org/get-started/locally/)
- visdom and tensorboardX if you want to use the visualisation of loss and evaluation
  -- if do want to use them set visdom/tensorboard flag equal to true while training 
  -- and configure the visdom port in arguments in  `train.py.`
- OpenCV is need 

## TRAINING
Please follow dataset preparation [README](https://github.com/gurkirt/FPN.pytorch/tree/master/prep) from `prep` folder of this repo.
Once you have pre-processed the dataset, then you are ready to train your networks.

To train run the following command. 

`python train.py --dataset=voc --basenet=resnet50 --batch_size=24 --lr=0.0005 -j=8  --ngpu=2 --step_values=40000 --max_iter=50000`

It will use all the visible GPUs. You can append `CUDA_VISIBLE_DEVICES=gpuids-comma-separated` at the beginning of the above command to mask certain GPUs. We used two GPU machine to run these experiments.

Also, check the arguments in `train.py` to adjust your training process.