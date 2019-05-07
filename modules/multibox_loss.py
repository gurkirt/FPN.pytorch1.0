import torch.nn as nn
import torch.nn.functional as F
import torch, pdb, time
from modules import box_utils
'''

Credits:: https://github.com/amdegroot/ssd.pytorch
& https://github.com/qfgaohao/pytorch-ssd/blob/master/vision/nn/multibox_loss.py

'''
class MultiBoxLoss(nn.Module):
    def __init__(self, neg_pos_ratio=3):
        """Implement SSD Multibox Loss.
        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiBoxLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, confidence, predicted_locations, gts, anchors):
        # cls_out, reg_out, anchor_gt_labels, anchor_gt_locations
        """


        Compute classification loss and smooth l1 loss.
        Args:
            confidence (batch_size, num_anchors, num_classes): class predictions.
            locations (batch_size, num_anchors, 4*seq_len): predicted locations.
            labels (batch_size, num_anchors): real labels of all the anchors.
            boxes (batch_size, num_anchors, 4*seq_len): real boxes corresponding all the anchors.


        """
        
        num_classes = confidence.size(2)
        gt_locations = []
        labels = []
        with torch.no_grad():
            # torch.cuda.synchronize()
            # t0 = time.perf_counter()
            for b in range(len(gts)):
                gt_boxes = gts[b][:,:4]
                gt_labels = gts[b][:,4]
                gt_labels = gt_labels.type(torch.cuda.LongTensor)

                conf, loc = box_utils.match_anchors(gt_boxes, gt_labels, anchors)

                labels.append(conf)
                gt_locations.append(loc)
            gt_locations = torch.stack(gt_locations, 0)
            labels = torch.stack(labels, 0)
            # torch.cuda.synchronize()
            # t1 = time.perf_counter()
            # print(gt_locations.size(), labels.size(), t1 - t0)
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)
        
        # pdb.set_trace()

        confidence = confidence[mask, :]
        classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], reduction='sum')
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')
        num_pos = pos_mask.sum()
        return smooth_l1_loss/num_pos, classification_loss/num_pos

class YOLOLoss(nn.Module):
    def __init__(self, neg_pos_ratio=3):
        """Implement SSD Multibox Loss.
        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(YOLOLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.bce_loss = nn.BCELoss().cuda()
        self.pos_weight = 1.0
        self.neg_weight = 100.0


    def forward(self, confidence, predicted_locations, gts, anchors):
        # cls_out, reg_out, anchor_gt_labels, anchor_gt_locations
        """

        Compute classification loss and smooth l1 loss.
        Args:
            confidence (batch_size, num_anchors, num_classes): class predictions.
            locations (batch_size, num_anchors, 4*seq_len): predicted locations.
            labels (batch_size, num_anchors): real labels of all the anchors.
            boxes (batch_size, num_anchors, 4*seq_len): real boxes corresponding all the anchors.

        """
        
        
        gt_locations = []
        labels = []
        with torch.no_grad():
            # torch.cuda.synchronize()
            # t0 = time.perf_counter()
            for b in range(len(gts)):
                gt_boxes = gts[b][:,:4]
                gt_labels = gts[b][:,4]
                gt_labels = gt_labels.type(torch.cuda.LongTensor)

                conf, loc = box_utils.match_anchors(gt_boxes, gt_labels, anchors)

                labels.append(conf)
                gt_locations.append(loc)
            gt_locations = torch.stack(gt_locations, 0)
            labels = torch.stack(labels, 0)
        
        # pdb.set_trace()
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        num_pos = pos_mask.sum()
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')/num_pos
        
        ## Compute classification loss
        confidence = torch.sigmoid(confidence)
        cls_conf = confidence[:,:,1:]
        binary_preds = confidence[:,:,0]
        num_classes = cls_conf.size(2)

        conf_labels = labels[pos_mask]-1
        y_onehot = cls_conf.new_zeros(conf_labels.size(0), num_classes)
        y_onehot[range(y_onehot.shape[0]), conf_labels] = 1.0
        classification_loss = self.bce_loss(cls_conf[pos_mask].reshape(-1, num_classes), y_onehot)

        labels_bin = (labels>0).float()
        neg_mask = labels<1
        binary_loss_pos = self.bce_loss(binary_preds[pos_mask], labels_bin[pos_mask])
        binary_loss_neg = self.bce_loss(binary_preds[neg_mask], labels_bin[neg_mask])
        binary_loss = binary_loss_pos*self.pos_weight + binary_loss_neg*self.neg_weight 

        return smooth_l1_loss, classification_loss + binary_loss
