""" 
    
    Adapted from: extract_features.py
    Modification by: Gurkirt Singh
    Modification started: 2nd April 2019
    large parts of this files are from many github repos
    mainly adopted from
    https://github.com/gurkirt/realtime-action-detection

    Please don't remove above credits and give star to these repos
    Licensed under The MIT License [see LICENSE for details]

    Purpose: Takes an input image and performs detections and plot them
    
"""

import os
import pdb
import time 
import json
import cv2
import socket
import getpass 
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from modules import utils
from modules.anchor_box_kmeans import anchorBox as kanchorBoxes
from modules.anchor_box_base import anchorBox
from modules.evaluation import evaluate_detections
from modules.box_utils import decode, nms
from modules.spatial_flatten import Spatialflatten
from modules import  AverageMeter
from data import BaseTransform
from models.fpn import build_fpn_unshared
from models.fpn_shared_heads import build_fpn_shared_heads
from modules.utils import get_class_names
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import random

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Run demo on a single image')
parser.add_argument('--anchor_type', default='kmeans', help='kmeans or default')
# Name of backbone networ, e.g. resnet18, resnet34, resnet50, resnet101 resnet152 are supported 
parser.add_argument('--basenet', default='resnet50', help='pretrained base model')
parser.add_argument('--dataset', default='coco', help='pretrained base model')
# if output heads are have shared features or not: 0 is no-shareing else sharining enabled
parser.add_argument('--shared_heads', default=1, type=int,help='0 mean no shareding more than 0 means shareing')
parser.add_argument('--bias_heads', default=0, type=int,help='0 mean no bias in head layears')
# Input size of image only 600 is supprted at the moment 
parser.add_argument('--input_dim', default=600, type=int, help='Input Size for SSD')
# Evaluation hyperparameters
parser.add_argument('--final_thresh', default=0.5, type=float, help='Confidence threshold to pick final detections ')
parser.add_argument('--conf_thresh', default=0.01, type=float, help='Confidence threshold before nms')
parser.add_argument('--nms_thresh', default=0.5, type=float, help='NMS threshold')
parser.add_argument('--num_nodes', default=50, type=int, help='manualseed for reproduction')
parser.add_argument('--topk', default=200, type=int, help='topk for evaluation')
# Program arguments
parser.add_argument('--start', default=0, type=int, help='topk for evaluation')
parser.add_argument('--end', default=200, type=int, help='topk for evaluation')

parser.add_argument('--ngpu', default=2, type=int, help='If  more than then use all visible GPUs by default only one GPU used ') 
# Use CUDA_VISIBLE_DEVICES=0,1,4,6 to selct GPUs to use
parser.add_argument('--model_path', default='/mnt/mars-gamma/coco/cache/FPN600-kmeanssh01-coco-bs24-resnet50-lr00050-bn0/model_150000.pth', help='Location to model weights') # /mnt/mars-fast/datasets/
parser.add_argument('--samples_path', default='demo_data/samples/', help='Location image samples') # /mnt/sun-gamma/datasets/
parser.add_argument('--outputs_path', default='demo_data/outputs/', help='Location where outputs would be stored') # /mnt/sun-gamma/datasets/

## Parse arguments
args = parser.parse_args()
random.seed(123)

torch.set_default_tensor_type('torch.FloatTensor')


def main(img_names):
    anchors = 'None'
    with torch.no_grad():
        if args.anchor_type == 'kmeans':
            anchorbox = kanchorBoxes(input_dim=args.input_dim, dataset=args.dataset)
        else:
            anchorbox = anchorBox(args.anchor_type, input_dim=args.input_dim, dataset=args.dataset)
        anchors = anchorbox.forward()
        args.ar = anchorbox.ar
    
    
    args.num_anchors = anchors.size(0)
    anchors = anchors.cuda(0, non_blocking=True)
    
    if args.dataset == 'coco':
        args.num_classes = 81
    else:
        args.num_classes = 21
    
    cmaps = plt.cm.get_cmap('jet', args.num_classes)
    bbox_colors = [cmaps(i) for i in range(args.num_classes)]
    random.shuffle(bbox_colors)
    args.classes = get_class_names(args.dataset)

    args.means =[0.485, 0.456, 0.406]
    args.stds = [0.229, 0.224, 0.225]
    transform = BaseTransform(args.input_dim, args.means, args.stds)
    args.bias_heads = args.bias_heads>0
    args.head_size = 256
    if args.shared_heads>0:
        net = build_fpn_shared_heads(args.basenet, '', ar=args.ar, head_size=args.head_size, num_classes=args.num_classes, bias_heads=args.bias_heads)
    else: 
        net = build_fpn_unshared(args.basenet, '', ar=args.ar, head_size=args.head_size, num_classes=args.num_classes, bias_heads=args.bias_heads)
    
    net = net.cuda()
    
    with torch.no_grad():

        if args.ngpu>1:
            print('\nLets do dataparallel\n')
            net = torch.nn.DataParallel(net)
        
        net.eval()
        net.load_state_dict(torch.load(args.model_path))
        
        softmax = nn.Softmax(dim=2).cuda().eval()
        for image_name in img_names:
            print(args.samples_path, image_name)
            img = np.asarray([cv2.imread(args.samples_path+image_name)])
            
            images, _ , _ = transform(img, [], [])
            images = torch.stack(images, 0)

            obj_boxes, cls_scores, bin_scores, obj_ids = extract_boxes(args, images, net, softmax, anchors)
            # pdb.set_trace()
            img = img[0]
            height,width,ch = img.shape
            fig, ax = plt.subplots(1)
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            print('Number of boxes detected', obj_boxes.shape[0])
            count = 0
            for ik in range(obj_boxes.size(0)):
                if bin_scores[ik]>args.final_thresh: ## only plot boxes that has higher score than this threshold
                    count += 1
                    scores = cls_scores[ik, 1:].squeeze().cpu().numpy()
                    win_class = np.argmax(scores)
                    win_label = args.classes[win_class]
                    box = obj_boxes[ik,:].cpu().numpy()
                    x1 = box[0] * width; x2 = box[2] * width
                    y1 = box[1] * height; y2 = box[3] * height
                    b_colour = bbox_colors[win_class]
                    bbox = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=b_colour, facecolor="none")
                    ax.add_patch(bbox)
                    plt.text(x1, y1, s=win_label, color="white", verticalalignment="top", bbox={"color": b_colour, "pad": 0})
                    
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            plt.savefig(args.outputs_path+image_name, bbox_inches="tight", pad_inches=0.0)
            plt.close()
            print('Number of object plotted', count)

def extract_boxes(args, images, net, softmax, anchors):

    
    images = images.cuda(0, non_blocking=True)
    
    loc_data, sf_conf = net(images)

    sf_conf = torch.softmax(sf_conf, 2)
    
    b=0
    
    decoded_boxes = decode(loc_data[b], anchors, [0.1, 0.2])
    obj_boxes, obj_ids, cls_scores, bin_scores = apply_nms(decoded_boxes, 1-sf_conf[b,:, 0], sf_conf[b], 10, args.conf_thresh, args.nms_thresh)

    return  obj_boxes, cls_scores, bin_scores, obj_ids

def apply_nms(boxes, scores, sf_conf, num_nodes, conf_thresh, nms_thresh):
    
    all_ids = torch.arange(0, scores.size(0), 1).cuda()
    
    c_mask = scores.gt(conf_thresh)  # greater than minmum threshold
    new_scores = scores[c_mask].squeeze()
    all_ids = all_ids[c_mask].squeeze()
    
    if new_scores.dim() == 0:
        all_ids = torch.arange(0, scores.size(0), dtype=torch.cuda.LongTensor)
        c_mask = scores.gt(conf_thresh*0.001)  # greater than minmum threshold
        new_scores = scores[c_mask].squeeze()
        all_ids = all_ids[c_mask].squeeze()
    
    # boxes = decoded_boxes.clone()
    l_mask = c_mask.unsqueeze(1).expand_as(boxes)
    boxes = boxes[l_mask].view(-1, 4)
    l_mask = c_mask.unsqueeze(1).expand_as(sf_conf)
    # pdb.set_trace()
    sf_conf = sf_conf[l_mask].view(-1, sf_conf.size(1))
    # idx of highest scoring and non-overlapping boxes per class
    ids, counts = nms(boxes, new_scores, nms_thresh, num_nodes*20) 
    
    news = ids[:counts]
    new_scores = new_scores[news]
    all_ids = all_ids[news]
    boxes = boxes[news]
    sf_conf = sf_conf[news]

    # pdb.set_trace()
    return boxes, all_ids, sf_conf, new_scores

if __name__ == '__main__':
    img_names = os.listdir(args.samples_path)
    img_names = [img for img in img_names if img.endswith('.jpg')]
    main(img_names)
