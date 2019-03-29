import torch.nn as nn
import torch.nn.functional as F
import torch, pdb, time
from modules import box_utils

class MultiBoxLoss(nn.Module):
    def __init__(self, neg_pos_ratio=3):
        """Implement SSD Multibox Loss.
        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiBoxLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, confidence, predicted_locations, gts, priors):
        # cls_out, reg_out, prior_gt_labels, prior_gt_locations
        """

        Compute classification loss and smooth l1 loss.
        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4*seq_len): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4*seq_len): real boxes corresponding all the priors.

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

                conf, loc = box_utils.match_priors(gt_boxes, gt_labels, priors)

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
