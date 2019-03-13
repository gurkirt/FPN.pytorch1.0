
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
import socket
import getpass 
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from modules.prior_box import PriorBox
import torch.utils.data as data_utils
from data import Detection, detection_collate, BaseTransform
from data.augmentations import Augmentation
from models.fpn import build_fpn
from modules.multibox_loss import MultiBoxLoss
from modules.joint_loss import JointLoss
import numpy as np
import time
from modules.evaluation import evaluate_detections
from modules.box_utils import decode, nms
from modules import  AverageMeter
from torch.optim.lr_scheduler import MultiStepLR
from data.coocDetection import CocoDetection

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

parser = argparse.ArgumentParser(description='Retinet with FPN as base with resnet Training')
parser.add_argument('--version', default='v1', help='layer')
parser.add_argument('--basenet', default='resnet101', help='pretrained base model')
parser.add_argument('--dataset', default='voc', help='pretrained base model')
parser.add_argument('--input_dim', default=600, type=int, help='Input Size for SSD')
parser.add_argument('--input_type', default='rgb', type=str, help='INput tyep default rgb can take flow as well')
parser.add_argument('--use_bg', default=False, type=str2bool, help='If to use bground frames')
parser.add_argument('--input_frames', default=1, type=int, help='Number of input frame, default for rgb is 1')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--max_iter', default=120000, type=int, help='Number of training iterations')
parser.add_argument('--eval_step', default=15000, type=int, help='Number of training iterations before evaluation')
parser.add_argument('--man_seed', default=1, type=int, help='manualseed for reproduction')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=1, type=int, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--step_values', default='60000,90000', type=str, help='Chnage the lr @')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--vis_port', default=8098, type=int, help='Port for Visdom Server')
parser.add_argument('--data_root', default='/mnt/mercury-fast/datasets/', help='Location to root directory fo dataset') # /mnt/mars-fast/datasets/
parser.add_argument('--save_root', default='/mnt/mercury-fast/datasets/', help='Location to save checkpoint models') # /mnt/sun-gamma/datasets/
parser.add_argument('--iou_thresh', default=0.5, type=float, help='Evaluation threshold')
parser.add_argument('--conf_thresh', default=0.01, type=float, help='Confidence threshold for evaluation')
parser.add_argument('--nms_thresh', default=0.45, type=float, help='NMS threshold')
parser.add_argument('--topk', default=1500, type=int, help='topk for evaluation')
## Focal loss parameters
parser.add_argument('--is_focal_loss', default=False, type=str2bool, help='is Focal loss being used or multibox loss')
parser.add_argument('--activation_type_softmax', default=False, type=str2bool, help='is Focal loss type is softmax')
parser.add_argument('--fl_alpha', default=0.25, type=float, help='Focal Loss alpha')
parser.add_argument('--fl_gamma', default=2.0, type=float, help='Focal loss gamma')
##verbosity
parser.add_argument('--v', default=True, type=str2bool, help='')


## Parse arguments
args = parser.parse_args()
hostname = socket.gethostname()

if hostname == 'mars':
    args.data_root = '/mnt/mars-fast/datasets/'
    args.save_root = '/mnt/mars-fast/datasets/'
    args.vis_port = 8097
elif hostname == 'alien':
    args.data_root = '/home/gurkirt/'
    args.save_root = '/home/gurkirt/'
    args.vis_port = 8099

if args.is_focal_loss:
    if args.activation_type_softmax:
        args.loss_type = 'FLsoftmax'
    else:
        args.loss_type = 'FLsigmoid'
else:
    args.loss_type = 'OHEM'
    args.activation_type_softmax = True

## set random seeds
np.random.seed(args.man_seed)
torch.manual_seed(args.man_seed)

if args.cuda:
    torch.cuda.manual_seed_all(args.man_seed)

torch.set_default_tensor_type('torch.FloatTensor')

NUM_CLASSES = {'ucf24': 24, 'daly': 10, 'voc': 20, 'coco': 80}


def main():
    if args.activation_type_softmax:
        args.num_classes = NUM_CLASSES[args.dataset] + 1
    else:
        args.num_classes = NUM_CLASSES[args.dataset] + 1

    args.step_values = [int(val) for val in args.step_values.split(',')]
    args.loss_reset_step = 30
    args.print_step = 10
    args.pr_th = 11

    ## Define the experiment Name will used to same directory and ENV for visdom
    args.action = False
    if args.dataset == 'ucf24' or args.dataset == 'daly':
        args.action = True
    if args.action:
        args.exp_name = 'FPN-{}-bg{}-{}-bs-{}-if-{}-{}-lr-{:05d}'.format(args.dataset, args.use_bg,
                                                                         args.input_type, args.batch_size,
                                                                         args.input_frames, args.basenet,
                                                                         int(args.lr * 100000))
    else:
        args.exp_name = 'FPN-{}-{}-bs-{}-{}-lr{:05d}'.format(args.dataset,args.loss_type,
                                                          args.batch_size,
                                                          args.basenet,
                                                          int(args.lr * 100000))

    args.save_root += args.dataset+'/'
    args.save_root = args.save_root+'cache/'+args.exp_name+'/'

    if not os.path.isdir(args.save_root):
        os.makedirs(args.save_root)

    priorbox = PriorBox(input_dim=args.input_dim, is_cuda=args.cuda)
    priors = Variable(priorbox.forward(), require_grad=False)
    args.num_priors = priors.size(0)
    args.ar = priorbox.anchor_boxes
    args.head_size = 256

    args.means =[0.485, 0.456, 0.406]
    args.stds = [0.229, 0.224, 0.225]
    net = build_fpn(modelname=args.basenet, ar=args.ar, head_size=args.head_size, num_classes=args.num_classes)
    net = net.cuda()
    if args.ngpu>1:
        print('\n\n\nLets do dataparallel\n\n\n\n')
        net = torch.nn.DataParallel(net)

    # parameter_dict = dict(net.named_parameters()) # Get parmeter of network in dictionary format wtih name being key
    # params = []
    # Set different learning rate to bias layers and set their weight_decay to 0
    # for name, param in parameter_dict.items():
    #     if name.find('bias') > -1:
    #         print(name, 'layer parameters will be trained @ {}'.format(args.lr*2))
    #         params += [{'params': [param], 'lr': args.lr*2, 'weight_decay': 0}]
    #     else:
    #         print(name, 'layer parameters will be trained @ {}'.format(args.lr))
    #         params += [{'params':[param], 'lr': args.lr, 'weight_decay':args.weight_decay}]


    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.is_focal_loss:
        criterion = JointLoss(num_classes=args.num_classes, overlap_thresh=args.jaccard_threshold, is_softmax=args.activation_type_softmax)
    else:
        criterion = MultiBoxLoss(num_classes=args.num_classes, overlap_thresh=args.jaccard_threshold)
    scheduler = MultiStepLR(optimizer, milestones=args.step_values, gamma=args.gamma)

    train(args, net, priors, optimizer, criterion, scheduler)


def train(args, net, priors, optimizer, criterion, scheduler):
    log_file = open(args.save_root+"training.log", "w", 1)
    log_file.write(args.exp_name+'\n')
    for arg in vars(args):
        print(arg, getattr(args, arg))
        log_file.write(str(arg)+': '+str(getattr(args, arg))+'\n')
    log_file.write(str(net))

    net.train()
    net.module.base_net.apply(set_bn_eval)

    # loss counters
    batch_time = AverageMeter()
    losses = AverageMeter()
    loc_losses = AverageMeter()
    cls_losses = AverageMeter()

    print('Loading Dataset...')
    train_dataset = Detection(args, 'train', Augmentation(args.input_dim, args.means, args.stds))
    # train_dataset = Detection(args, 'train', BaseTransform(args.input_dim, args.means, args.stds))
    log_file.write(train_dataset.print_str)
    print('TRAIN-DATA :::>>>\n',train_dataset.print_str)
    val_dataset = Detection(args, 'test', BaseTransform(args.input_dim, args.means, args.stds), full_test=False)
    log_file.write(val_dataset.print_str)
    print('VAL-DATA :::>>>\n', val_dataset.print_str)
    epoch_size = len(train_dataset) // args.batch_size
    print('Training RetinaNet on', train_dataset.dataset)

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
        for cls in train_dataset.classes:
            legends.append(cls)
        print(legends)
        val_lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1,args.num_classes)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='AP %',
                title='Current Validation APs and mAP',
                legend=legends
            )
        )


    batch_iterator = None
    train_data_loader = data_utils.DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate)
    val_data_loader = data_utils.DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers,
                                 shuffle=False, collate_fn=detection_collate, pin_memory=True)
    itr_count = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iteration = 0
    while iteration <= args.max_iter:
        for i, (images, targets, img_indexs) in enumerate(train_data_loader):
            if iteration > args.max_iter:
                break
            iteration += 1
            # load train data
            #images, targets, img_indexs = next(batch_iterator)
            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
            else:
                images = Variable(images)
                targets = [Variable(anno, volatile=True) for anno in targets]
            # forward
            out = net(images)
            # reset gradient buffers
            optimizer.zero_grad()
            # computer loss
            loss_l, loss_c = criterion(out, targets, priors)
            loss = loss_l + loss_c
            # backprop
            loss.backward()
            # optimisation step
            optimizer.step()
            scheduler.step()
            loc_loss = loss_l.data[0]
            conf_loss = loss_c.data[0]*100
            # print('Loss data type ',type(loc_loss))
            if iteration > args.pr_th:
                if loc_loss>1000:
                    lline = '\n\n\n We got faulty location loss {} {} \n\n\n'.format(loc_loss, conf_loss)
                    log_file.write(lline)
                    print(lline)
                    loc_loss = 20.0
                if conf_loss>100000:
                    lline = '\n\n\n We got faulty classification loss {} {} \n\n\n'.format(loc_loss, conf_loss)
                    log_file.write(lline)
                    print(lline)
                    conf_loss = 20.0
                loc_losses.update(loc_loss)
                cls_losses.update(conf_loss)
                losses.update((loc_loss + conf_loss)/2.0)

            if iteration % args.print_step == 0 and iteration > args.pr_th+1:
                if args.visdom:
                    losses_list = [loc_losses.val, cls_losses.val, losses.val, loc_losses.avg, cls_losses.avg, losses.avg]
                    viz.line(X=torch.ones((1, 6)).cpu() * iteration,
                        Y=torch.from_numpy(np.asarray(losses_list)).unsqueeze(0).cpu(),
                        win=lot,
                        update='append')


                torch.cuda.synchronize()
                t1 = time.perf_counter()
                batch_time.update(t1 - t0)

                print_line = 'Itration {:06d}/{:06d} loc-loss {:.3f}({:.3f}) cls-loss {:.5f}({:.5f}) ' \
                             'average-loss {:.3f}({:.3f}) Timer {:0.3f}({:0.3f})'.format(
                              iteration, args.max_iter, loc_losses.val, loc_losses.avg, cls_losses.val,
                              cls_losses.avg, losses.val, losses.avg, batch_time.val, batch_time.avg)

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                log_file.write(print_line+'\n')
                print(print_line)

                # if args.visdom and args.send_images_to_visdom:
                #     random_batch_index = np.random.randint(images.size(0))
                #     viz.image(images.data[random_batch_index].cpu().numpy())
                itr_count += 1

                if itr_count % args.loss_reset_step == 0 and itr_count > 0:
                    loc_losses.reset()
                    cls_losses.reset()
                    losses.reset()
                    batch_time.reset()
                    print('Reset ', args.exp_name,' after', itr_count*args.print_step)
                    itr_count = 0


            if (iteration % args.eval_step == 0 or iteration == 5000) and iteration>0:
                torch.cuda.synchronize()
                tvs = time.perf_counter()
                print('Saving state, iter:', iteration)
                torch.save(net.state_dict(), args.save_root+'ssd300_ucf24_' +
                           repr(iteration) + '.pth')

                net.eval() # switch net to evaluation mode
                mAP, ap_all, ap_strs = validate(args, net, val_data_loader, val_dataset, priors, iteration, iou_thresh=args.iou_thresh)

                for ap_str in ap_strs:
                    print(ap_str)
                    log_file.write(ap_str+'\n')
                ptr_str = '\nMEANAP:::=>'+str(mAP)+'\n'
                print(ptr_str)
                log_file.write(ptr_str)

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


def validate(args, net, val_data_loader, val_dataset, prior_data, iteration_num, iou_thresh=0.5):
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
    activation = nn.Sigmoid().cuda()
    if args.activation_type_softmax:
        activation = nn.Softmax().cuda()
    for val_itr, (images, targets, img_indexs) in enumerate(val_data_loader):

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        batch_size = images.size(0)
        height, width = images.size(2), images.size(3)

        if args.cuda:
            images = Variable(images.cuda(), volatile=True)
        output = net(images)

        loc_data = output[0]
        conf_preds = output[1]


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
            decoded_boxes = decode(loc_data[b].data, prior_data.data, [0.1, 0.2]).clone()
            conf_scores = activation(conf_preds[b]).data.clone()

            for cl_ind in range(1, num_classes):
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
        if val_itr%val_step == 0:
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
