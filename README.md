# Training single stage Feature Pyramid Network (FPN) with online hard example mining (OHEM)

# It is getting ready, another couple of days: updated on 29th March
# train.py works waiting for results on VOC 

We freeze the batch normlisation layers of  
## Performance map@0.5 ResNet18

Hand picked anchors per scale 9: 74.2

Hand picked anchors per scale 3: 72.8

Kmean 3 anchors per scale: 74.1

## Performance map@0.5 ResNet50

Kmean 3 anchors per scale: 79.7
 
## Results will be out soon on coco as well

## TRAINING
Please follow dataset prepration [README](https://github.com/gurkirt/FPN.pytorch/tree/master/prep) from `prep` folder of this repo.
Once you have pre-processed the dataset then you are ready to train your networks.

To train run the following command. 

`python train.py --dataset=voc --basenet=resnet50 --batch_size=24 --lr=0.0005 -j=8  --ngpu=2 --step_values=40000 --max_iter=50000`

It will use all the visiable GPUs. You can append `CUDA_VISIBLE_DEVICES=gpuids-comma-seprated` at the begning of above command to mask certain GPUs. We used two GPU machince to run these experiments.

Also, check the agruments in `train.py` to adjust your training process.