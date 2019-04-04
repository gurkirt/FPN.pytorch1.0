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
import sys
import time
import socket
import getpass 
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from modules import utils
from modules.anchor_box_kmeans import anchorBox as kanchorBoxes
from modules.anchor_box_base import anchorBox
from modules.multibox_loss import MultiBoxLoss
from modules.evaluation import evaluate_detections
from modules.box_utils import decode, nms
from modules import  AverageMeter
from data import Detection, BaseTransform, custum_collate
from models.fpn import build_fpn
from train import validate

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Training single stage FPN with OHEM, resnet as backbone')
# anchor_type to be used in the experiment
parser.add_argument('--anchor_type', default='kmeans', help='kmeans or default')
# Name of backbone networ, e.g. resnet18, resnet34, resnet50, resnet101 resnet152 are supported 
parser.add_argument('--basenet', default='resnet101', help='pretrained base model')
#  Name of the dataset only voc or coco are supported
parser.add_argument('--dataset', default='voc', help='pretrained base model')
# Input size of image, only 600 is supprted at the moment 
parser.add_argument('--input_dim', default6300, type=int, help='Input Size for SSD')
#  data loading argumnets
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--num_workers', '-j', default=4, type=int, help='Number of workers used in dataloading')
# optimiser hyperparameters
parser.add_argument('--max_iter', default=150000, type=int, help='Number of training iterations')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--step_values', default='60000,90000', type=str, help='Chnage the lr @')
# Freeze batch normlisatio layer or not 
parser.add_argument('--bn', default=0, type=int, help='if 0 freeze or else keep updating bn layers')

# Evaluation hyperparameters
parser.add_argument('--model_path', default='', type=str, help='Path to model directory')
parser.add_argument('--eval_iters', default='150000,', type=str, help='Chnage the lr @')
parser.add_argument('--val_step', default=10000, type=int, help='Number of training iterations before evaluation')
parser.add_argument('--iou_thresh', default=0.5, type=float, help='Evaluation threshold')
parser.add_argument('--conf_thresh', default=0.001, type=float, help='Confidence threshold for evaluation')
parser.add_argument('--nms_thresh', default=0.45, type=float, help='NMS threshold')
parser.add_argument('--topk', default=50, type=int, help='topk for evaluation')

# Progress logging
parser.add_argument('--log_iters', default=True, type=str2bool, help='Print the loss at each iteration')
parser.add_argument('--log_step', default=10, type=int, help='Log after k steps for text/Visdom/tensorboard')

# Program arguments
parser.add_argument('--man_seed', default=1, type=int, help='manualseed for reproduction')
parser.add_argument('--ngpu', default=1, type=int, help='If  more than then use all visible GPUs by default only one GPU used ') 
# Use CUDA_VISIBLE_DEVICES=0,1,4,6 to selct GPUs to use
parser.add_argument('--data_root', default='/mnt/mercury-fast/datasets/', help='Location to root directory fo dataset') # /mnt/mars-fast/datasets/
parser.add_argument('--save_root', default='/mnt/mercury-fast/datasets/', help='Location to save checkpoint models') # /mnt/sun-gamma/datasets/

## Parse arguments
args = parser.parse_args()

import socket
import getpass
username = getpass.getuser()
hostname = socket.gethostname()
args.hostname = hostname
args.user = username
args.model_dir = args.data_root
print('\n\n ', username, ' is using ', hostname, '\n\n')

if username == 'gurkirt':
    args.model_dir = '/mnt/mars-gamma/global-models/pytorch-imagenet/'
    if hostname == 'mars':
        args.data_root = '/mnt/mars-fast/datasets/'
        args.save_root = '/mnt/mars-gamma/'
        args.vis_port = 8097
    elif hostname in ['sun','jupiter']:
        args.data_root = '/mnt/mercury-fast/datasets/'
        args.save_root = '/mnt/mars-gamma/'
        if hostname in ['sun']:
            args.vis_port = 8096
        else:
            args.vis_port = 8095
    elif hostname == 'mercury':
        args.data_root = '/mnt/mercury-fast/datasets/'
        args.save_root = '/mnt/mars-gamma/'
        args.vis_port = 8098
    elif hostname.startswith('comp'):
        args.data_root = '/home/gurkirt/datasets/'
        args.save_root = '/home/gurkirt/cache/'
        args.vis_port = 8097
        visdom=False
        args.model_dir = args.data_root+'weights/'
    else:
        raise('ERROR!!!!!!!! Specify directories')


## set random seeds and global settings
np.random.seed(args.man_seed)
torch.manual_seed(args.man_seed)
torch.cuda.manual_seed_all(args.man_seed)
torch.set_default_tensor_type('torch.FloatTensor')


def main():

    args.eval_iters = [int(val) for val in args.eval_iters.split(',')]
    # args.loss_reset_step = 10
    args.log_step = 10
    args.dataset = args.dataset.lower()
    args.basenet = args.basenet.lower()
    
    args.bn = abs(args.bn) # 0 freeze or else use bn
    if args.bn>0:
        args.bn = 1 # update bn layer set the flag to 1

    args.exp_name = 'FPN{:d}-{:s}-{:s}-bs{:02d}-{:s}-lr{:05d}-bn{:d}'.format(args.input_dim, args.anchor_type, args.dataset,
                                                          args.batch_size,
                                                          args.basenet,
                                                          int(args.lr * 100000),
                                                          args.bn)

    args.save_root += args.dataset+'/'
    args.save_root = args.save_root+'cache/'+args.exp_name+'/'

    # Should already be created during training time
    if not os.path.isdir(args.save_root): # if save directory doesn't exist create it
        os.makedirs(args.save_root)

    source_dir = args.save_root+'/source/' # where to save the source
    utils.copy_source(source_dir)

    anchors = 'None'
    with torch.no_grad():
        if args.anchor_type == 'kmeans':
            anchorbox = kanchorBoxes(input_dim=args.input_dim, dataset=args.dataset)
        else:
            anchorbox = anchorBox(args.anchor_type, input_dim=args.input_dim, dataset=args.dataset)
        anchors = anchorbox.forward()
        args.ar = anchorbox.ar
    
    args.num_anchors = anchors.size(0)

    if args.dataset == 'coco':
        args.train_sets = ['train2017']
        args.val_sets = ['val2017']
    else:
        args.train_sets = ['train2007', 'val2007', 'train2012', 'val2012']
        args.val_sets = ['test2007']

    args.means =[0.485, 0.456, 0.406]
    args.stds = [0.229, 0.224, 0.225]
    val_dataset = Detection(args, train=False, image_sets=args.val_sets, 
                            transform=BaseTransform(args.input_dim, args.means, args.stds), full_test=False)
    print('Done Loading Dataset Validation Dataset :::>>>\n',val_dataset.print_str)
    
    args.num_classes = len(val_dataset.classes) + 1
    args.classes = val_dataset.classes
    
    args.head_size = 256

    net = build_fpn(args.basenet, args.model_dir, ar=args.ar, head_size=args.head_size, num_classes=args.num_classes)
    net = net.cuda()

    if args.ngpu>1:
        print('\nLets do dataparallel\n')
        net = torch.nn.DataParallel(net)
    net.eval()

    for iteration in args.eval_iters:
        log_file = open("{:s}/testing-{:d}.log".format(args.save_root, iteration), "w", 1)
        log_file.write(args.exp_name + '\n')
        if len(args.model_path)<5:
            args.model_path = args.save_root + '/model_' + repr(iteration) + '.pth'
        log_file.write(args.model_path+'\n')
    
        net.load_state_dict(torch.load(args.model_path))

        print('Finished loading model %d !' % iteration)
        # Load dataset
        val_data_loader = data_utils.DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers,
                                 shuffle=False, pin_memory=True, collate_fn=custum_collate)

        # evaluation
        torch.cuda.synchronize()
        tt0 = time.perf_counter()
        log_file.write('Testing net \n')
        net.eval() # switch net to evaluation mode
        mAP, ap_all, ap_strs , det_boxes = validate(args, net, anchors, val_data_loader, val_dataset, iteration, iou_thresh=args.iou_thresh)

        for ap_str in ap_strs:
            print(ap_str)
            log_file.write(ap_str+'\n')
        ptr_str = '\nMEANAP:::=>'+str(mAP)+'\n'
        print(ptr_str)
        log_file.write(ptr_str)

        torch.cuda.synchronize()
        print('Complete set time {:0.2f}'.format(time.perf_counter() - tt0))
        log_file.close()

if __name__ == '__main__':
    main()
