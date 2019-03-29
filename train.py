
""" Adapted from:

    Modification by: Gurkirt Singh
    Modification started: 13th March

    Parts of this files are from many github repos
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Which was adopated by: Ellis Brown, Max deGroot
    https://github.com/amdegroot/ssd.pytorch

    Futher updates from 
    https://github.com/qfgaohao/pytorch-ssd
    https://github.com/gurkirt/realtime-action-detection

    maybe more but that is where I got these from
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
import shutil
import torch.nn as nn
import torch.optim as optim
from modules.anchor_box_kmeans import anchorBox
# from modules.anchor_box_base import anchorBox
import torch.utils.data as data_utils
from data import Detection, BaseTransform, custum_collate
from data.augmentations import Augmentation
from models.fpn import build_fpn
from modules.multibox_loss import MultiBoxLoss
from modules.evaluation import evaluate_detections
from modules.box_utils import decode, nms
from modules import  AverageMeter
from torch.optim.lr_scheduler import MultiStepLR


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description='Retinet with FPN as base with resnet Training')
# version or add name to experiment
parser.add_argument('--version', default='kmeansAR3', help='layer')
# Name of backbone networ, e.g. resnet18, resnet34, resnet50, resnet101 resnet152 are supported 
parser.add_argument('--basenet', default='resnet101', help='pretrained base model')
#  Name of the dataset only voc or coco are supported
parser.add_argument('--dataset', default='voc', help='pretrained base model')
# Input size of image onlye 300 is supprted at the moment 
parser.add_argument('--input_dim', default=300, type=int, help='Input Size for SSD')
#  data loading argumnets
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--num_workers', '-j', default=4, type=int, help='Number of workers used in dataloading')
# optimiser hyperparameters
parser.add_argument('--max_iter', default=150000, type=int, help='Number of training iterations')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--step_values', default='60000,90000', type=str, help='Chnage the lr @')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')

# Loss function matching threshold
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')

# Evaluation hyperparameters
parser.add_argument('--val_step', default=10000, type=int, help='Number of training iterations before evaluation')
parser.add_argument('--iou_thresh', default=0.5, type=float, help='Evaluation threshold')
parser.add_argument('--conf_thresh', default=0.001, type=float, help='Confidence threshold for evaluation')
parser.add_argument('--nms_thresh', default=0.45, type=float, help='NMS threshold')
parser.add_argument('--topk', default=50, type=int, help='topk for evaluation')

# Progress logging
parser.add_argument('--log_iters', default=True, type=str2bool, help='Print the loss at each iteration')
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

print('\n\n ', username, ' is using ', hostname, '\n\n')
if username == 'gurkirt':
    if hostname == 'mars':
        args.data_root = '/mnt/mars-fast/datasets/'
        args.save_root = '/mnt/mercury-alpha/'
        args.vis_port = 8097
    elif hostname in ['sun','jupiter']:
        args.data_root = '/mnt/mercury-fast/datasets/'
        args.save_root = '/mnt/mercury-alpha/'
        if hostname in ['sun']:
            args.vis_port = 8096
        else:
            args.vis_port = 8095
    elif hostname == 'mercury':
        args.data_root = '/mnt/mercury-fast/datasets/'
        args.save_root = '/mnt/mercury-alpha/'
        args.vis_port = 8098
    elif hostname.startswith('comp'):
        args.data_root = '/home/gurkirt/datasets/'
        args.save_root = '/home/gurkirt/cache/'
        args.vis_port = 8097
        visdom=False
    else:
        raise('ERROR!!!!!!!! Specify directories')

if args.tensorboard:
    from tensorboardX import SummaryWriter

## set random seeds and global settings
np.random.seed(args.man_seed)
torch.manual_seed(args.man_seed)
torch.cuda.manual_seed_all(args.man_seed)
torch.set_default_tensor_type('torch.FloatTensor')

# Freeze batch normlisation layers
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def main():

    args.step_values = [int(val) for val in args.step_values.split(',')]
    args.loss_reset_step = 10
    args.print_step = 10
    args.dataset = args.dataset.lower()
    args.basenet = args.basenet.lower()

    args.exp_name = 'FPN{:d}-{:s}-{:s}-bs{:02d}-{:s}-lr{:05d}'.format(args.input_dim, args.version, args.dataset,
                                                          args.batch_size,
                                                          args.basenet,
                                                          int(args.lr * 100000))

    args.save_root += args.dataset+'/'
    args.save_root = args.save_root+'cache/'+args.exp_name+'/'

    if not os.path.isdir(args.save_root): # if save directory doesn't exist create it
        os.makedirs(args.save_root)

    source_dir = args.save_root+'/source/'
    if not os.path.isdir(source_dir):
        os.system('mkdir -p ' + source_dir)
    
    for dirpath, dirs, files in os.walk('./', topdown=True):
        for file in files:
            if file.endswith('.py'): #fnmatch.filter(files, filepattern):
                shutil.copy2(os.path.join(dirpath, file), source_dir)

    anchors = 'None'
    with torch.no_grad():
        anchorbox = anchorBox(input_dim=args.input_dim, dataset=args.dataset)
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

    print('\nLoading Datasets')
    train_dataset = Detection(args, train=True, image_sets=args.train_sets, 
                            transform=Augmentation(args.input_dim, args.means, args.stds))
    print('Done Loading Dataset Train Dataset :::>>>\n',train_dataset.print_str)
    val_dataset = Detection(args, train=False, image_sets=args.val_sets, 
                            transform=BaseTransform(args.input_dim, args.means, args.stds), full_test=False)
    print('Done Loading Dataset Validation Dataset :::>>>\n',val_dataset.print_str)
    
    args.num_classes = len(train_dataset.classes) + 1
    args.classes = train_dataset.classes
    
    args.head_size = 256
    
    net = build_fpn(args.basenet, args.data_root, ar=args.ar, head_size=args.head_size, num_classes=args.num_classes)
    net = net.cuda()
    
    if args.ngpu>1:
        print('\nLets do dataparallel\n')
        net = torch.nn.DataParallel(net)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    criterion = MultiBoxLoss()
    
    scheduler = MultiStepLR(optimizer, milestones=args.step_values, gamma=args.gamma)

    train(args, net, anchors, optimizer, criterion, scheduler, train_dataset, val_dataset)


def train(args, net, anchors, optimizer, criterion, scheduler, train_dataset, val_dataset):
    
    anchors = anchors.cuda(0, non_blocking=True)
    if args.tensorboard:
        log_dir = args.save_root+'tensorboard-{date:%m-%d-%Hx}.log'.format(date=datetime.datetime.now())
        sw = SummaryWriter(log_dir=log_dir)
    log_file = open(args.save_root+'training.text{date:%m-%d-%Hx}.txt'.format(date=datetime.datetime.now()), 'w', 1)
    log_file.write(args.exp_name+'\n')
    # log_file.write(sys.argv)
    for arg in vars(args):
        print(arg, getattr(args, arg))
        log_file.write(str(arg)+': '+str(getattr(args, arg))+'\n')
    log_file.write(str(net))

    net.train()
    net.module.base_net.apply(set_bn_eval)

    # loss counters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loc_losses = AverageMeter()
    cls_losses = AverageMeter()

    # train_dataset = Detection(args, 'train', BaseTransform(args.input_dim, args.means, args.stds))
    log_file.write(train_dataset.print_str)
    log_file.write(val_dataset.print_str)
    print('Train-DATA :::>>>', train_dataset.print_str)
    print('VAL-DATA :::>>>', val_dataset.print_str)
    epoch_size = len(train_dataset) // args.batch_size
    print('Training FPN on ', train_dataset.dataset,'\n')

    if args.visdom:
        import visdom
        viz = visdom.Visdom()
        viz.port = args.vis_port
        viz.env = args.exp_name
        # initialize visdom loss plot
        lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 6)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title='Current Training Loss',
                legend=['REG', 'CLS', 'AVG', 'S-REG', ' S-CLS', ' S-AVG']
            )
        )
        # initialize visdom meanAP and class APs plot
        legends = ['meanAP']
        for cls_ in args.classes:
            legends.append(cls_)
        print(legends)
        val_lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, args.num_classes)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='AP %',
                title='Current Validation APs and mAP',
                legend=legends
            )
        )


    train_data_loader = data_utils.DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers,
                                  shuffle=True, pin_memory=True, collate_fn=custum_collate )
    val_data_loader = data_utils.DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers,
                                 shuffle=False, pin_memory=True, collate_fn=custum_collate)
    itr_count = 0
    torch.cuda.synchronize()
    start = time.perf_counter()
    iteration = 0

    while iteration <= args.max_iter:
        for i, (images, gts, _) in enumerate(train_data_loader):
            if iteration > args.max_iter:
                break
            iteration += 1
            images = images.cuda(0, non_blocking=True)
            gts = [anno.cuda(0, non_blocking=True) for anno in gts]
            # forward
            torch.cuda.synchronize()
            data_time.update(time.perf_counter() - start)
            reg_out, cls_out = net(images)

            optimizer.zero_grad()
            loss_l, loss_c = criterion(cls_out, reg_out, gts, anchors)
            loss = loss_l + loss_c

            loss.backward()
            optimizer.step()
            scheduler.step()

            # pdb.set_trace()
            loc_loss = loss_l.item()
            conf_loss = loss_c.item()
            
            if loc_loss>1000:
                lline = '\n\n\n We got faulty LOCATION loss {} {} \n\n\n'.format(loc_loss, conf_loss)
                log_file.write(lline)
                print(lline)
                loc_loss = 20.0
            if conf_loss>1000:
                lline = '\n\n\n We got faulty CLASSIFICATION loss {} {} \n\n\n'.format(loc_loss, conf_loss)
                log_file.write(lline)
                print(lline)
                conf_loss = 20.0
            # print('Loss data type ',type(loc_loss))
            loc_losses.update(loc_loss)
            cls_losses.update(conf_loss)
            losses.update((loc_loss + conf_loss)/2.0)

            torch.cuda.synchronize()
            batch_time.update(time.perf_counter() - start)
            start = time.perf_counter()

            if iteration % args.print_step == 0 and iteration > 11:
                if args.visdom:
                    losses_list = [loc_losses.val, cls_losses.val, losses.val, loc_losses.avg, cls_losses.avg, losses.avg]
                    viz.line(X=torch.ones((1, 6)).cpu() * iteration,
                        Y=torch.from_numpy(np.asarray(losses_list)).unsqueeze(0).cpu(),
                        win=lot,
                        update='append')
                if args.tensorboard:
                    sw.add_scalars('Classification', {'val': cls_losses.val, 'avg':cls_losses.avg},iteration)
                    sw.add_scalars('Localisation', {'val': loc_losses.val, 'avg':loc_losses.avg},iteration)
                    sw.add_scalars('Overall', {'val': losses.val, 'avg':losses.avg},iteration)
                    
                print_line = 'Itration {:06d}/{:06d} loc-loss {:.2f}({:.2f}) cls-loss {:.2f}({:.2f}) ' \
                             'average-loss {:.2f}({:.2f}) DataTime{:0.2f}({:0.2f}) Timer {:0.2f}({:0.2f})'.format(
                              iteration, args.max_iter, loc_losses.val, loc_losses.avg, cls_losses.val,
                              cls_losses.avg, losses.val, losses.avg, 10*data_time.val, 10*data_time.avg, 10*batch_time.val, 10*batch_time.avg)

                log_file.write(print_line+'\n')
                print(print_line)
                itr_count += 1

                if itr_count % args.loss_reset_step == 0 and itr_count > 0:
                    loc_losses.reset()
                    cls_losses.reset()
                    losses.reset()
                    batch_time.reset()
                    data_time.reset()
                    print('Reset ', args.exp_name,' after', itr_count*args.print_step)
                    itr_count = 0


            if (iteration % args.val_step == 0 or iteration == 5000) and iteration>10:
                torch.cuda.synchronize()
                tvs = time.perf_counter()
                print('Saving state, iter:', iteration)
                torch.save(net.state_dict(), args.save_root+'FPN_model_' +
                           repr(iteration) + '.pth')

                net.eval() # switch net to evaluation mode
                mAP, ap_all, ap_strs = validate(args, net, anchors, val_data_loader, val_dataset, iteration, iou_thresh=args.iou_thresh)

                for ap_str in ap_strs:
                    print(ap_str)
                    log_file.write(ap_str+'\n')
                ptr_str = '\nMEANAP:::=>'+str(mAP)+'\n'
                print(ptr_str)
                log_file.write(ptr_str)

                if args.tensorboard:
                    sw.add_scalar('mAP@0.5', mAP, iteration)
                    class_AP_group = dict()
                    for c, ap in enumerate(ap_all):
                        class_AP_group[args.classes[c]] = ap
                    sw.add_scalars('ClassAPs', class_AP_group, iteration)

                if args.visdom:
                    aps = [mAP]
                    for ap in ap_all:
                        aps.append(ap)
                    viz.line(
                        X=torch.ones((1, args.num_classes)).cpu() * iteration,
                        Y=torch.from_numpy(np.asarray(aps)).unsqueeze(0).cpu(),
                        win=val_lot,
                        update='append'
                            )
                net.train() # Switch net back to training mode
                net.module.base_net.apply(set_bn_eval)

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                prt_str = '\nValidation TIME::: {:0.3f}\n\n'.format(t0-tvs)
                print(prt_str)
                log_file.write(ptr_str)

    log_file.close()


def validate(args, net, anchors,  val_data_loader, val_dataset, iteration_num, iou_thresh=0.5):
    """Test a FPN network on an image database."""
    print('Validating at ', iteration_num)
    num_images = len(val_dataset)
    num_classes = args.num_classes
    
    det_boxes = [[] for _ in range(num_classes-1)]
    gt_boxes = []
    print_time = True
    val_step = 5
    count = 0
    torch.cuda.synchronize()
    ts = time.perf_counter()
    softmax = nn.Softmax(dim=2).cuda()
    with torch.no_grad():
        for val_itr, (images, targets, img_indexs) in enumerate(val_data_loader):

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
                    boxes = boxes[ids[:counts]].cpu().numpy()
                    # print('boxes sahpe',boxes.shape)
                    boxes[:,0] *= width
                    boxes[:,2] *= width
                    boxes[:,1] *= height
                    boxes[:,3] *= height

                    for ik in range(boxes.shape[0]):
                        boxes[ik, 0] = max(0, boxes[ik, 0])
                        boxes[ik, 2] = min(width, boxes[ik, 2])
                        boxes[ik, 1] = max(0, boxes[ik, 1])
                        boxes[ik, 3] = min(height, boxes[ik, 3])

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

    print('Evaluating detections for itration number ', iteration_num)
    return evaluate_detections(gt_boxes, det_boxes, val_dataset.classes, iou_thresh=iou_thresh)


if __name__ == '__main__':
    main()
