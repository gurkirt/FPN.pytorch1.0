import torch, pdb
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .box_utils import match, log_sum_exp

class JointLoss(nn.Module):
    """FPN Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Focal Loss to handle excessive number of negative examples
           that comes with using a large number of default bounding boxes.
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh = 0.5, neg_overlap = 0.4, use_gpu=True, is_softmax=True):
        super(JointLoss, self).__init__()
        self.use_gpu = use_gpu
        self.is_softmax = is_softmax
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]
        self.bg_threshold = 0.4
        self.tp_threshold = 0.5
        self.alpha = 0.25
        self.gamma = 2.0
        self.eps = 1e-07

    def forward(self, predictions, targets, priors):
        """Joint Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from FPN net.
                conf shape: torch.size(batch_size, num_anchors,num_classes)
                loc shape: torch.size(batch_size,num_anchors,4)
                priors shape: torch.size(num_anchors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data = predictions

        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        alpha = self.alpha
        gamma = self.gamma
        # match priors (default boxes) and ground truth boxes
        if self.use_gpu:
            loc_t = torch.cuda.FloatTensor(num, num_priors, 4)
            loc_mask = torch.cuda.ByteTensor(num, num_priors)
            conf_t = torch.cuda.LongTensor(num, num_priors)
            # conf_mask = torch.cuda.LongTensor(num,num_priors)
        else:
            loc_t = torch.FloatTensor(num, num_priors, 4)
            loc_mask = torch.ByteTensor(num, num_priors)
            conf_t = torch.LongTensor(num, num_priors)
            # conf_mask = torch.LongTensor(num, num_priors)
        # pdb.set_trace()
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match(self.bg_threshold, self.tp_threshold, truths, defaults, self.variance, labels, loc_t, conf_t, loc_mask, idx)

        ## Location regression
        conf_t = Variable(conf_t, requires_grad=False)
        pos = conf_t>0
        N = pos.data.sum()
        N1 = torch.sum(conf_t.data > -1)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        # loc_mask = Variable(loc_mask, requires_grad=False)
        # loc_mask = loc_mask.unsqueeze(loc_mask.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = Variable(loc_t, requires_grad=False)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False) / N

        # Compute max conf across batch for hard negative mining
        #

        if self.is_softmax:
            pos1 = conf_t > -1
            N = pos1.data.sum()
            # print(pos.size())
            # pdb.set_trace()
            pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(conf_data)
            conf_data = conf_data[pos_idx1].view(-1, self.num_classes)
            log_p = F.log_softmax(conf_data)
            conf_t = conf_t[conf_t > -1].view(-1,1)
            log_p = log_p.gather(1, conf_t)
            log_p = log_p.view(-1, 1)
            p = Variable(log_p.data.exp(), require_grad=False)
            loss_c = -1 * pow(1 - p, self.gamma) * log_p

        else:
            conf_data = conf_data.view(-1, self.num_classes)
            conf_t = conf_t.view(-1, 1).squeeze(1)
            t = self._one_hot_embedding(conf_t.data, self.num_classes)
            if self.use_gpu:
                t = Variable(t.cuda()).view(-1, self.num_classes)
            else:
                t = Variable(t).view(-1, self.num_classes)

            # N1 = torch.sum(conf_t > 0)
            conf_t = conf_t.unsqueeze(conf_t.dim()).expand_as(t)
            t = t[conf_t > -1].view(-1, self.num_classes)
            logits = conf_data[conf_t > -1].view(-1, self.num_classes)
            # xt = logits * (2 * t - 1)  # xt = x if t > 0 else -x
            # pt = (gamma * xt + 1).sigmoid()
            # # w = alpha * t + (1 - alpha) * (1 - t)
            # loss_c = -1*pt.log() / gamma
            p = F.sigmoid(logits)
            pt = p * t + (1 - p) * (1 - t)
            pt = pt.clamp(self.eps, 1. - self.eps)# pt = p if t > 0 else 1-p
            alpha_t = alpha * t + (1 - alpha) * (1 - t)  # w = alpha if t > 0 else 1-alpha
            w = alpha_t * ((1 - pt).pow(gamma))
            loss_c = - w * torch.log(pt) #F.binary_cross_entropy_with_logits(logits, t, w, size_average=False)

        loss_c = loss_c.sum()
        loss_c /= N1
        # print('l and c ', loss_l.data[0], loss_c.data[0], N1, N)
        return loss_l, loss_c

    def _one_hot_embedding(self, conf_t, num_classes):
        conf_t[conf_t<1] = 0
        y = torch.eye(num_classes).cuda() # [D,D]
        # y = y[:, 1:]
        return y[conf_t,:]
