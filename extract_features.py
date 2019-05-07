""" 
    
    Adapted from:
    Modification by: Gurkirt Singh
    Modification started: 2nd April 2019
    large parts of this files are from many github repos
    mainly adopted from
    https://github.com/gurkirt/realtime-action-detection

    Please don't remove above credits and give star to these repos
    Licensed under The MIT License [see LICENSE for details]
    
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

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='FPN feature extraction')
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
parser.add_argument('--conf_thresh', default=0.0001, type=float, help='Confidence threshold for evaluation')
parser.add_argument('--nms_thresh', default=0.6, type=float, help='NMS threshold')
parser.add_argument('--num_nodes', default=50, type=int, help='manualseed for reproduction')
parser.add_argument('--topk', default=200, type=int, help='topk for evaluation')
# Program arguments
parser.add_argument('--start', default=0, type=int, help='topk for evaluation')
parser.add_argument('--end', default=200, type=int, help='topk for evaluation')

parser.add_argument('--ngpu', default=2, type=int, help='If  more than then use all visible GPUs by default only one GPU used ') 
# Use CUDA_VISIBLE_DEVICES=0,1,4,6 to selct GPUs to use
parser.add_argument('--model_path', default='/home/gurkirt/cache/coco/cache/FPN600-kmeanssh01-coco-bs24-resnet50-lr00050-bn0/model_150000.pth', help='Location to model weights') # /mnt/mars-fast/datasets/
parser.add_argument('--base_dir', default='/home/gurkirt/datasets/vidvrd/', help='Location to save checkpoint models') # /mnt/sun-gamma/datasets/

## Parse arguments
args = parser.parse_args()

torch.set_default_tensor_type('torch.FloatTensor')


def main(allimages):
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
    feature_size = [75, 38, 19, 10, 5]
    kd = 3
    args.kd = kd
    log_file = open("{:s}/extracting_s{:d}_e{:d}.log".format(args.base_dir, args.start, args.end), "w", 1)
    # log_file.write(args.exp_name + '\n')
    log_file.write(args.model_path+'\n')
    
    with torch.no_grad():
        flatten_layers = nn.ModuleList([Spatialflatten(feature_size[0]**2,  kd=kd), 
                                        Spatialflatten(feature_size[1]**2,  kd=kd), 
                                        Spatialflatten(feature_size[2]**2,  kd=kd), 
                                        Spatialflatten(feature_size[3]**2,  kd=kd), 
                                        Spatialflatten(feature_size[4]**2,  kd=kd)])
        flatten_layers = flatten_layers.cuda().eval()
        
        if args.ngpu>1:
            print('\nLets do dataparallel\n')
            net = torch.nn.DataParallel(net)
        
        net.eval()
        net.load_state_dict(torch.load(args.model_path))
        
        softmax = nn.Softmax(dim=2).cuda().eval()
        last_id = torch.LongTensor([anchors.size(0) + args.ar -1]).cuda()
        empty_box = torch.zeros(1,4).cuda()
        empty_score = torch.zeros(1,args.num_classes).cuda()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        print_num = 10
        num_all_images = len(allimages)
        for idx, info in enumerate(allimages):
            save_path = info[1]
            save_filename = save_path + info[2][:-4] + '.pth'
            if not os.path.isfile(save_filename): # only compute features for images for which feature are not computed yet
                path_img = info[0] + info[2]
                img = np.asarray([cv2.imread(path_img)])
                images, _ , _ = transform(img, [], [])
                images = torch.stack(images, 0)
    
                node_features, boxes, cls_scores, conf_scores = extract_feat(args, images, net, softmax, flatten_layers, anchors, args.num_nodes, last_id, empty_box, empty_score)
                
                if not os.path.isdir(save_path):
                    os.mkdir(save_path)

                if idx % print_num == 0:
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    pt_str = 'Done [{:03d}/{:d}] times taken for {:d} images is {:0.1f}'.format(idx, num_all_images, print_num, t1-t0)
                    log_file.write(pt_str+'\n')
                    print(pt_str)
                    t0 = t1

                fboxes = boxes.cpu().squeeze(0)
                fscores = conf_scores.cpu().view(-1, 1)
                frame_feats = torch.cat((fscores, fboxes), 1)
                frame_feats = torch.cat((frame_feats, cls_scores.cpu().squeeze(0)), 1)
                frame_feats = torch.cat((frame_feats, node_features.cpu().squeeze(0)), 1)
                frame_feats = frame_feats.type(torch.FloatTensor)

                # print(frame_feats.size(), type(frame_feats))
                torch.save(frame_feats, save_filename)

                # torch.save({'node_features':node_features.cpu(), 'boxes':boxes.cpu(), 'cls_scores':cls_scores, 'conf_scores':conf_scores}, save_filename)
                # print(save_filename)

def extract_feat(args, images, net, softmax, flatten_layers, anchors, num_nodes, last_id, empty_box, empty_score):

    
    images = images.cuda(0, non_blocking=True)
    
    loc_data, conf_data, features = net(images, get_features=True)
    
    conf_scores_all = softmax(conf_data)
    flattened_features = list()
    for x, l in zip(features, flatten_layers):
        flattened_features.append(l(x))
    
    flattened_features.append(torch.zeros(flattened_features[0].size(0), 1, args.head_size*args.kd*args.kd).cuda())
    
    flat_features = flattened_features[0]
    for k in range(len(flattened_features)-1):
            flat_features = torch.cat((flat_features, flattened_features[k+1]),1)
    

    ids, boxes, cls_scores, conf_scores = get_ids_n_boxes(loc_data, conf_scores_all, anchors, num_nodes, last_id, empty_box, empty_score)
    # pdb.set_trace()
    node_features = get_features(ids, flat_features, args.ar)

    return node_features, boxes, cls_scores, conf_scores

def get_features(ids, features, ar):
    poolled_fetaures = []
    for b in range(len(ids)):
        indexs = ids[b]//ar
        # print(indexs)
        poolled_fetaures.append(features[b].index_select(0, indexs))
        # print(poolled_fetaures[b].size())
    
    return torch.stack(poolled_fetaures, 0)


def get_ids_n_boxes(loc, sf_conf, anchors, num_nodes, last_id, empty_box, empty_score):
    ids = []
    boxes = []; 
    all_scores = [];
    scores = []
    for b in range(loc.size(0)):
        decoded_boxes = decode(loc[b], anchors, [0.1, 0.2])
        obj_boxes, obj_ids, obj_scores, bin_scores = apply_nms(decoded_boxes, 1-sf_conf[b,:, 0], sf_conf[b], num_nodes, args.conf_thresh, args.nms_thresh)
        # pdb.set_trace()
        num_obj = obj_ids.size(0)
        if num_nodes>num_obj:
            # print(obj_ids.size(), obj_boxes.size(), self.last_id.size(), self.empty_box.size(), self.last_id)
            for _ in range(num_nodes - num_obj):
                obj_ids = torch.cat((obj_ids, last_id), 0)
                obj_boxes = torch.cat((obj_boxes, empty_box), 0)
                obj_scores = torch.cat((obj_scores, empty_score), 0)
                bin_scores = torch.cat((obj_scores, empty_score[0]), 0)
        else:
            obj_ids = obj_ids[:num_nodes]
            obj_boxes = obj_boxes[:num_nodes]
            obj_scores = obj_scores[:num_nodes]
            bin_scores = bin_scores[:num_nodes]
        # print(obj_ids.size(), obj_boxes.size())
        ids.append(obj_ids)
        boxes.append(obj_boxes)
        all_scores.append(obj_scores)
        scores.append(bin_scores)

    return ids, torch.stack(boxes, 0), torch.stack(all_scores, 0),  torch.stack(scores, 0)

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

    allimages = []
    allcount = 0
    annots = dict()

    downloaded = os.listdir(args.base_dir+'rgb-images/')
    downloaded = reversed(sorted(downloaded))
    # downloaded = downloaded[args.start:args.end]
    # print('number of videos are', len(downloaded))
    for itr, vid in enumerate(downloaded):
        viddir = args.base_dir + 'rgb-images/' + vid + '/'
        images = os.listdir(viddir)
        save_dir = args.base_dir + 'features_new/' + vid + '/'
        name = vid.split('_')[1] 
        subset = 'train'
        annot_file = args.base_dir + subset +'/'+ vid + '.json'
        if not os.path.isfile(annot_file):
            subset = 'test'
            annot_file = args.base_dir + subset +'/'+ vid + '.json'

        with open(annot_file, 'r') as f:
            annos = json.load(f)
        # pdb.set_trace()
        gt_num_images = annos['frame_count']
        num_images = len(images)
        ptstr = vid + ' has ' +  str(num_images) + 'images and gt has ' + str(gt_num_images)
        print(ptstr)
        assert gt_num_images == num_images, ptstr

        for d in sorted(images):
            if d.endswith('.jpg'):
                allimages.append([viddir, save_dir, d])

    print('Total number images to be used for feature extraction ', len(allimages))
    main(allimages)
