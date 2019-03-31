# Single stage Feature Pyramid Network (FPN) with online hard example mining (OHEM)

This repostory contain single stage version of FPN present on [RetinaNet paper](https://arxiv.org/pdf/1708.02002.pdf).
Objective to reproduce Table 1 with ResNet50 with OHEM.

## Archtecture 

![RetinaNet Structure](/figures/retinaNet.png)

ResNet is used as backbone network (a) to build the pyramid features (b). 
Each classifciation (c) and regression (d) subnet is made of 4 convolutional layers and finally a convolutional layer to predict the class scores and bounding box coordinated respectovely.

## Loss function 
We use multibox loss function with online hard example mining (OHEM), similar to [SSD](https://arxiv.org/pdf/1512.02325.pdf).
A huge thanks to Max deGroot, Ellis Brown for [Pytorch implementation](https://github.com/amdegroot/ssd.pytorch) of SSD and loss function.


## Anchors
We use two types of anchor.
### Pre-defined anchors
Similar to RetinaNet, we can build anchor or sometime called prior boxes in SSD.
As a baseline we 3 aspect ratios (ar) and 3 scale ratios (sr) per pyramid-level.
Which results in 9 anchors per cell location in the grid of each pyramid level.
Resulting total number of anchors/predications close to `67K`

OR, we can have only one scale ratio. Reasoning behind that is the pyramid should be able to capture the scale space. 
Now, we will have 3 anchors per cell location in the grid of each pyramid level.
Resulting total number of anchors/predications close to `22K`. 
Although we will save the computional cost this way by predicting less boxes, but the recall of result anchor boxes drops drastically.

### K-mean the anchors
Alternative is to boost the recall of one scale ratio by performing k-means on ground truth boxes with k=3.
We pick 3 anchors from each pyramid level and intial cluster center and then perform cluster on ground truth boxes.
Intersection-over-union (IoU) is used as distance metric. 
Since the cluster centers and centered around origin, we need to move the center of each ground truth box to origin as well.

We performed clustering for `coco` and `voc` 

# It is getting ready, another couple of days: updated on 29th March

We freeze the batch normlisation layers of  

## Performance map@0.5 ResNet50

Kmean 3 anchors per scale: 79.7
 
## Results will be out soon on coco as well

## Installation
- We used anaconda 3.7 as python distribution
- You will need [Pytorch1.0](https://pytorch.org/get-started/locally/)
- visdom and tensorboardX if you want to use them visulisation of loss and evaluetion
  -- if do want to use them set visdom/tensorboard flag equal to true while training 
  -- visdom port as well see agguments for `train.py`
- Opencv is need 

## TRAINING
Please follow dataset prepration [README](https://github.com/gurkirt/FPN.pytorch/tree/master/prep) from `prep` folder of this repo.
Once you have pre-processed the dataset then you are ready to train your networks.

To train run the following command. 

`python train.py --dataset=voc --basenet=resnet50 --batch_size=24 --lr=0.0005 -j=8  --ngpu=2 --step_values=40000 --max_iter=50000`

It will use all the visiable GPUs. You can append `CUDA_VISIBLE_DEVICES=gpuids-comma-seprated` at the begning of above command to mask certain GPUs. We used two GPU machince to run these experiments.

Also, check the agruments in `train.py` to adjust your training process.