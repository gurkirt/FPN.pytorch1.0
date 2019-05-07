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
import time, json
import socket
import getpass 
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torch.optim.lr_scheduler import MultiStepLR
from modules import utils
from modules.anchor_box_kmeans import anchorBox as kanchorBoxes
from modules.anchor_box_base import anchorBox
from modules.multibox_loss import MultiBoxLoss
from modules.evaluation import evaluate_detections
from modules.box_utils import decode, nms
from modules import  AverageMeter
from data import Detection, BaseTransform, custum_collate
from data.augmentations import Augmentation
from models.fpn import build_fpn_unshared
from models.fpn_shared_heads import build_fpn_shared_heads
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from train import validate

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description='Training single stage FPN with OHEM, resnet as backbone')
# anchor_type to be used in the experiment
parser.add_argument('--anchor_type', default='kmeans', help='kmeans or default')
# Name of backbone networ, e.g. resnet18, resnet34, resnet50, resnet101 resnet152 are supported 
parser.add_argument('--basenet', default='resnet101', help='pretrained base model')
# if output heads are have shared features or not: 0 is no-shareing else sharining enabled
parser.add_argument('--shared_heads', default=0, type=int,help='0 mean no shareding more than 0 means shareing')
parser.add_argument('--bias_heads', default=0, type=int,help='0 mean no bias in head layears')
#  Name of the dataset only voc or coco are supported
parser.add_argument('--dataset', default='voc', help='pretrained base model')
# Input size of image only 600 is supprted at the moment 
parser.add_argument('--input_dim', default=600, type=int, help='Input Size for SSD')
#  data loading argumnets
parser.add_argument('--batch_size', default=24, type=int, help='Batch size for training')
# Number of worker to load data in parllel
parser.add_argument('--num_workers', '-j', default=2, type=int, help='Number of workers used in dataloading')
# optimiser hyperparameters
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--eval_iters', default='150000', type=str, help='Chnage the lr @')

# Freeze batch normlisatio layer or not 
parser.add_argument('--bn', default=0, type=int, help='if 0 freeze or else keep updating bn layers')
# Loss function matching threshold
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')

# Evaluation hyperparameters
parser.add_argument('--iou_thresh', default=0.5, type=float, help='Evaluation threshold')
parser.add_argument('--conf_thresh', default=0.001, type=float, help='Confidence threshold for evaluation')
parser.add_argument('--nms_thresh', default=0.5, type=float, help='NMS threshold')
parser.add_argument('--topk', default=200, type=int, help='topk for evaluation')

# Progress logging
parser.add_argument('--log_iters', default=True, type=str2bool, help='Print the loss at each iteration')
parser.add_argument('--log_step', default=10, type=int, help='Log after k steps for text/Visdom/tensorboard')
parser.add_argument('--tensorboard', default=False, type=str2bool, help='Use tensorboard for loss/evalaution visualization')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom for loss/evalaution visualization')
parser.add_argument('--vis_port', default=8098, type=int, help='Port for Visdom Server')

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
        args.data_root = '/mnt/mars-fast/datasets/'
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

if args.tensorboard:
    from tensorboardX import SummaryWriter

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

    args.exp_name = 'FPN{:d}-{:s}sh{:02d}-{:s}-bs{:02d}-{:s}-lr{:05d}-bn{:d}'.format(args.input_dim, 
                                                          args.anchor_type, 
                                                          args.shared_heads, 
                                                          args.dataset,
                                                          args.batch_size,
                                                          args.basenet,
                                                          int(args.lr * 100000),
                                                          args.bn)

    args.save_root += args.dataset+'/'
    args.save_root = args.save_root+'cache/'+args.exp_name+'/'

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
    anchors = anchors.cuda(0, non_blocking=True)
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
    args.data_dir = val_dataset.root
    args.num_classes = len(val_dataset.classes) + 1
    args.classes = val_dataset.classes
    args.bias_heads = args.bias_heads>0
    args.head_size = 256
    if args.shared_heads>0:
        net = build_fpn_shared_heads(args.basenet, args.model_dir, ar=args.ar, head_size=args.head_size, num_classes=args.num_classes, bias_heads=args.bias_heads)
    else: 
        net = build_fpn_unshared(args.basenet, args.model_dir, ar=args.ar, head_size=args.head_size, num_classes=args.num_classes, bias_heads=args.bias_heads)
    
    net = net.cuda()

    if args.ngpu>1:
        print('\nLets do dataparallel\n')
        net = torch.nn.DataParallel(net)
    net.eval()

    for iteration in args.eval_iters:
        args.det_itr = iteration
        log_file = open("{:s}/testing-{:d}.log".format(args.save_root, iteration), "w", 1)
        log_file.write(args.exp_name + '\n')
        
        args.model_path = args.save_root + '/model_' + repr(iteration) + '.pth'
        log_file.write(args.model_path+'\n')
    
        net.load_state_dict(torch.load(args.model_path))

        print('Finished loading model %d !' % iteration)
        # Load dataset
        val_data_loader = data_utils.DataLoader(val_dataset, int(args.batch_size/2), num_workers=args.num_workers,
                                 shuffle=False, pin_memory=True, collate_fn=custum_collate)

        # evaluation
        torch.cuda.synchronize()
        tt0 = time.perf_counter()
        log_file.write('Testing net \n')
        net.eval() # switch net to evaluation mode
        if args.dataset != 'coco':
            mAP, ap_all, ap_strs , det_boxes = validate(args, net, anchors, val_data_loader, val_dataset, iteration, iou_thresh=args.iou_thresh)
        else:
            mAP, ap_all, ap_strs , det_boxes = validate_coco(args, net, anchors, val_data_loader, val_dataset, iteration, iou_thresh=args.iou_thresh)

        for ap_str in ap_strs:
            print(ap_str)
            log_file.write(ap_str+'\n')
        ptr_str = '\nMEANAP:::=>'+str(mAP)+'\n'
        print(ptr_str)
        log_file.write(ptr_str)

        torch.cuda.synchronize()
        print('Complete set time {:0.2f}'.format(time.perf_counter() - tt0))
        log_file.close()


def validate_coco(args, net, anchors,  val_data_loader, val_dataset, iteration_num, iou_thresh=0.5):
    """Test a FPN network on an image database."""
    print('Validating at ', iteration_num)

    annFile='{}/instances_{}.json'.format(args.data_dir,args.val_sets[0])
    cocoGT=COCO(annFile)
    coco_dets = []
    resFile = args.save_root + 'detections-{:05d}.json'.format(args.det_itr)
    resFile_txt = open(args.save_root + 'detections-{:05d}.txt'.format(args.det_itr), 'w')
    num_images = len(val_dataset)
    num_classes = args.num_classes
    
    det_boxes = [[] for _ in range(num_classes-1)]
    gt_boxes = []
    print_time = True
    val_step = 20
    count = 0
    torch.cuda.synchronize()
    ts = time.perf_counter()
    softmax = nn.Softmax(dim=2).cuda()
    idlist = val_dataset.idlist
    all_ids = val_dataset.ids

    with torch.no_grad():
        for val_itr, (images, targets, img_indexs, wh) in enumerate(val_data_loader):
            # if val_itr>1:
            #     break
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            batch_size = images.size(0)
            height, width = images.size(2), images.size(3)

            images = images.cuda(0, non_blocking=True)
            loc_data, conf_data = net(images)

            conf_scores_all = softmax(conf_data).clone()

            if print_time and val_itr%val_step == 0:
                torch.cuda.synchronize()
                tf = time.perf_counter()
                print('Forward Time {:0.3f}'.format(tf-t1))
            for b in range(batch_size):
                
                coco_image_id = int(all_ids[img_indexs[b]][1][8:])
                width, height = wh[b][0], wh[b][1]
                gt = targets[b].numpy()
                gt[:,0] *= width
                gt[:,2] *= width
                gt[:,1] *= height
                gt[:,3] *= height
                gt_boxes.append(gt)
                decoded_boxes = decode(loc_data[b], anchors, [0.1, 0.2]).clone()
                conf_scores = conf_scores_all[b].clone()
                #Apply nms per class and obtain the results
                for cl_ind in range(1, num_classes):
                    # pdb.set_trace()
                    scores = conf_scores[:, cl_ind].squeeze()
                    c_mask = scores.gt(args.conf_thresh)  # greater than minmum threshold
                    scores = scores[c_mask].squeeze()
                    # print('scores size',scores.size())
                    if scores.dim() == 0:
                        # print(len(''), ' dim ==0 ')
                        det_boxes[cl_ind - 1].append(np.asarray([]))
                        continue
                    boxes = decoded_boxes.clone()
                    l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                    boxes = boxes[l_mask].view(-1, 4)
                    # idx of highest scoring and non-overlapping boxes per class
                    ids, counts = nms(boxes, scores, args.nms_thresh, args.topk)  # idsn - ids after nms
                    scores = scores[ids[:counts]].cpu().numpy()
                    pick = min(scores.shape[0], 20)
                    scores = scores[:pick]
                    boxes = boxes[ids[:counts]].cpu().numpy()
                    boxes = boxes[:pick, :]
                    # print('boxes sahpe',boxes.shape)
                    boxes[:,0] *= width
                    boxes[:,2] *= width
                    boxes[:,1] *= height
                    boxes[:,3] *= height
                    cls_id = cl_ind-1
                    if len(idlist)>0:
                        cls_id = idlist[cl_ind-1]
                    # pdb.set_trace()
                    for ik in range(boxes.shape[0]):
                        boxes[ik, 0] = max(0, boxes[ik, 0])
                        boxes[ik, 2] = min(width, boxes[ik, 2])
                        boxes[ik, 1] = max(0, boxes[ik, 1])
                        boxes[ik, 3] = min(height, boxes[ik, 3])
                        box_ = [round(boxes[ik, 0], 1), round(boxes[ik, 1],1), round(boxes[ik, 2],1), round(boxes[ik, 3], 1)]
                        box_[2] = round(box_[2] - box_[0], 1)
                        box_[3] = round(box_[3] - box_[1], 1)
                        box_ = [float(b) for b in box_]
                        coco_dets.append({"image_id" : int(coco_image_id), "category_id" : int(cls_id), 
                                          "bbox" : box_, "score" : float(scores[ik]),
                                        })

                    cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=True)
                    det_boxes[cl_ind-1].append(cls_dets)
                count += 1
            if print_time and val_itr%val_step == 0:
                torch.cuda.synchronize()
                te = time.perf_counter()
                print('im_detect: {:d}/{:d} time taken {:0.3f}'.format(count, num_images, te-ts))
                torch.cuda.synchronize()
                ts = time.perf_counter()
            if print_time and val_itr%val_step == 0:
                torch.cuda.synchronize()
                te = time.perf_counter()
                print('NMS stuff Time {:0.3f}'.format(te - tf))
    
    # print('Evaluating detections for itration number ', iteration_num)

    mAP, ap_all, ap_strs , det_boxes = evaluate_detections(gt_boxes, det_boxes, val_dataset.classes, iou_thresh=iou_thresh)
    
    for ap_str in ap_strs:
        print(ap_str)
        resFile_txt.write(ap_str+'\n')
    ptr_str = '\nMEANAP:::=>'+str(mAP)+'\n'
    print(ptr_str)
    resFile_txt.write(ptr_str)
    
    print('saving results :::::')
    with open(resFile,'w') as f:
        json.dump(coco_dets, f)
    
    cocoDt=cocoGT.loadRes(resFile)
    # running evaluation
    cocoEval = COCOeval(cocoGT, cocoDt, 'bbox')
    # cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    resFile_txt.write(ptr_str)
    # pdb.set_trace()

    ptr_str = ''
    for s in cocoEval.stats:
        ptr_str += str(s) + '\n'
    print('\n\nPrintning COCOeval Generated results\n\n ')
    print(ptr_str)
    resFile_txt.write(ptr_str)
    return mAP, ap_all, ap_strs , det_boxes
    
if __name__ == '__main__':
    main()
