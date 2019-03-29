#Feature Pyramid Network (FPN) with online hard example mining (OHEM)

# It is not ready yet, another couple of days: updated on 19th March
# train.py works waiting for results on VOC 
## Performance map@0.5 ResNet18: with batch normlisation on
Hand picked anchors per scale 6: 74.2
Hand picked anchors per scale 3: 72.8
Kmean 3 anchors per scale: 74.1

##  Resnet50
Kmean 3 anchors per scale: 79.7

## Freezing the batch normlisation should work better 
## Results will be out soon on coco as well