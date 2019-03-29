import torch, pdb, math
import numpy as np

class Matchanchor(object):
    def __init__(self, anchors, variances=[0.1, 0.2], seq_len=1, iou_threshold=0.5):
        self.anchors = anchors.clone()
        # self.anchors_point_form = anchors.clone()
        # # pdb.set_trace()
        # for s in range(seq_len):
        #     self.anchors_point_form[:,s*4:(s+1)*4] = point_form(anchors[:,s*4:(s+1)*4])
        self.variances = variances
        self.seq_len = seq_len
        self.iou_threshold = iou_threshold


    def __call__(self, gt_boxes, gt_labels, num_mt=1):
            # pdb.set_trace()
            # pdb.set_trace()
            # num_mt = =len(gt_labels)
            if type(gt_boxes) is np.ndarray:
                gt_boxes = torch.from_numpy(gt_boxes)
            if type(gt_labels) is np.ndarray:
                gt_labels = torch.from_numpy(gt_labels)
            # pdb.set_trace()
            seq_overlaps =[]
            inds = torch.LongTensor([m*self.seq_len for m in range(num_mt)])  
            # print(inds, num_mt)
            ## get indexes of first frame in seq for each microtube
            gt_labels = gt_labels[inds]
            for s in range(self.seq_len):
                seq_overlaps.append(jaccard(gt_boxes[inds+s, :], point_form(self.anchors[:, s*4:(s+1)*4])))

            overlaps = seq_overlaps[0]
            ## Compute average overlap
            for s in range(self.seq_len-1):
                overlaps = overlaps + seq_overlaps[s+1]
            overlaps = overlaps/float(self.seq_len)
            # (Bipartite Matching)
            # [1,num_objects] best anchor for each ground truth
            best_anchor_overlap, best_anchor_idx = overlaps.max(1, keepdim=True)
            # [1,num_anchors] best ground truth for each anchor
            best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
            best_truth_idx.squeeze_(0)
            best_truth_overlap.squeeze_(0)
            best_anchor_idx.squeeze_(1)
            best_anchor_overlap.squeeze_(1)
            best_truth_overlap.index_fill_(0, best_anchor_idx, 2)  # ensure best anchor
            # TODO refactor: index  best_anchor_idx with long tensor
            # ensure every gt matches with its anchor of max overlap
            for j in range(best_anchor_idx.size(0)):
                best_truth_idx[best_anchor_idx[j]] = j

            conf = gt_labels[best_truth_idx] + 1         # Shape: [num_anchors]
            conf[best_truth_overlap < self.iou_threshold] = 0  # label as background

            for s in range(self.seq_len):
                st = gt_boxes[inds + s, :]
                matches = st[best_truth_idx]  # Shape: [num_anchors,4]
                if s == 0:
                    loc = encode(matches, self.anchors[:, s * 4:(s + 1) * 4], self.variances)  
                                # Shape: [num_anchors, 4] -- encode the gt boxes for frame i
                else:
                    temp = encode(matches, self.anchors[:, s * 4:(s + 1) * 4], self.variances)
                    loc = torch.cat([loc, temp], 1)  # shape: [num_anchors x 4 * seql_len] : stacking the location targets for different frames

            return conf, loc


def match_anchors(gt_boxes, gt_labels, anchors, iou_threshold=0.5, variances=[0.1, 0.2], seq_len=1):
            # pdb.set_trace()
            # pdb.set_trace()
            num_mt = int(gt_labels.size(0)/seq_len)
            
            # pdb.set_trace()
            seq_overlaps =[]
            inds = torch.LongTensor([m*seq_len for m in range(num_mt)])  
            # print(inds, num_mt)
            ## get indexes of first frame in seq for each microtube
            gt_labels = gt_labels[inds]
            for s in range(seq_len):
                seq_overlaps.append(jaccard(gt_boxes[inds+s, :], point_form(anchors[:, s*4:(s+1)*4])))

            overlaps = seq_overlaps[0]
            ## Compute average overlap
            for s in range(seq_len-1):
                overlaps = overlaps + seq_overlaps[s+1]
            overlaps = overlaps/float(seq_len)
            # (Bipartite Matching)
            # [1,num_objects] best anchor for each ground truth
            best_anchor_overlap, best_anchor_idx = overlaps.max(1, keepdim=True)
            # [1,num_anchors] best ground truth for each anchor
            best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
            best_truth_idx.squeeze_(0)
            best_truth_overlap.squeeze_(0)
            best_anchor_idx.squeeze_(1)
            best_anchor_overlap.squeeze_(1)
            best_truth_overlap.index_fill_(0, best_anchor_idx, 2)  # ensure best anchor
            # ensure every gt matches with its anchor of max overlap
            for j in range(best_anchor_idx.size(0)):
                best_truth_idx[best_anchor_idx[j]] = j

            conf = gt_labels[best_truth_idx] + 1         # Shape: [num_anchors]
            conf[best_truth_overlap < iou_threshold] = 0  # label as background

            for s in range(seq_len):
                st = gt_boxes[inds + s, :]
                matches = st[best_truth_idx]  # Shape: [num_anchors,4]
                if s == 0:
                    loc = encode(matches, anchors[:, s * 4:(s + 1) * 4], variances)  
                                # Shape: [num_anchors, 4] -- encode the gt boxes for frame i
                else:
                    temp = encode(matches, anchors[:, s * 4:(s + 1) * 4], variances)
                    loc = torch.cat([loc, temp], 1)  # shape: [num_anchors x 4 * seql_len] : stacking the location targets for different frames

            return conf, loc

def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.
    Args:
        loss (N, num_anchors): the loss for each example.
        labels (N, num_anchors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask
    
def point_form(boxes):
    """ Convert anchor_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from anchorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert anchor_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    # pdb.set_trace()
    # print(box_a.type(), box_b.type())
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) anchor boxes from anchorbox layers, Shape: [num_anchors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    # pdb.set_trace()
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def get_ovlp_cellwise(overlaps):
    feature_maps = [38, 19, 10, 5, 3, 1]
    aratios = [4, 6, 6, 6, 4, 4]
    dim = 0
    for f in feature_maps:
        dim += f*f
    out_ovlp = np.zeros(dim)
    count = 0
    st = 0
    for k, f in enumerate(feature_maps):
        ar = aratios[k]
        for i in range(f*f):
            et = st+ar
            ovlps_tmp = overlaps[0, st:et]
            #pdb.set_trace()
            out_ovlp[count] = max(ovlps_tmp)
            count += 1
            st = et
    assert count == dim

    return out_ovlp


def encode(matched, anchors, variances):
    """Encode the variances from the anchorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the anchor boxes.
    Args:
        matched: (tensor) Coords of ground truth for each anchor in point-form
            Shape: [num_anchors, 4].
        anchors: (tensor) anchor boxes in center-offset form
            Shape: [num_anchors,4].
        variances: (list[float]) Variances of anchorboxes
    Return:
        encoded boxes (tensor), Shape: [num_anchors, 4]
    """

    # dist b/t match center and anchor's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - anchors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * anchors[:, 2:])
    # match wh / anchor wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / anchors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_anchors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, anchors, variances):
    """Decode locations from predictions using anchors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_anchors,4]
        anchors (tensor): anchor boxes in center-offset form.
            Shape: [num_anchors,4].
        variances: (list[float]) Variances of anchorboxes
    Return:
        decoded bounding box predictions
    """
    #pdb.set_trace()
    boxes = torch.cat((
        anchors[:, :2] + loc[:, :2] * variances[0] * anchors[:, 2:],
        anchors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode_seq(loc, anchors, variances, seq_len):
    boxes = []
    #print('variances', variances)
    for s in range(seq_len):
        if s == 0:
            boxes = decode(loc[:, :4], anchors[:, :4], variances)
        else:
            boxes = torch.cat((boxes,decode(loc[:,s*4:(s+1)*4], anchors[:,s*4:(s+1)*4], variances)),1)

    return boxes

def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=20):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_anchors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_anchors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_anchors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep, 0
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count
