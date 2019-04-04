import torch
from math import sqrt as sqrt
from itertools import product as product
import numpy as np

anchor_boxes_kmeaned = dict()
anchor_boxes_kmeaned['voc'] = np.asarray( [ [0.0000, 0.0000, 0.0433, 0.1438],
                                            [0.0000, 0.0000, 0.0398, 0.0629],
                                            [0.0000, 0.0000, 0.0983, 0.0683],
                                            [0.0000, 0.0000, 0.0736, 0.2678],
                                            [0.0000, 0.0000, 0.0875, 0.1306],
                                            [0.0000, 0.0000, 0.2076, 0.1170],
                                            [0.0000, 0.0000, 0.1426, 0.4120],
                                            [0.0000, 0.0000, 0.1401, 0.2035],
                                            [0.0000, 0.0000, 0.2647, 0.2684],
                                            [0.0000, 0.0000, 0.2297, 0.6208],
                                            [0.0000, 0.0000, 0.3709, 0.4549],
                                            [0.0000, 0.0000, 0.6085, 0.2921],
                                            [0.0000, 0.0000, 0.4404, 0.7740],
                                            [0.0000, 0.0000, 0.8366, 0.8730],
                                            [0.0000, 0.0000, 0.7577, 0.5387]])

anchor_boxes_kmeaned['coco'] = np.asarray( [[0.0000, 0.0000, 0.0303, 0.0847],
                                            [0.0000, 0.0000, 0.0178, 0.0297],
                                            [0.0000, 0.0000, 0.0636, 0.0395],
                                            [0.0000, 0.0000, 0.0578, 0.2030],
                                            [0.0000, 0.0000, 0.0771, 0.0975],
                                            [0.0000, 0.0000, 0.1852, 0.0742],
                                            [0.0000, 0.0000, 0.1184, 0.3661],
                                            [0.0000, 0.0000, 0.1385, 0.1778],
                                            [0.0000, 0.0000, 0.3055, 0.1610],
                                            [0.0000, 0.0000, 0.2054, 0.6091],
                                            [0.0000, 0.0000, 0.2756, 0.3349],
                                            [0.0000, 0.0000, 0.6240, 0.2592],
                                            [0.0000, 0.0000, 0.4125, 0.6808],
                                            [0.0000, 0.0000, 0.8336, 0.8634],
                                            [0.0000, 0.0000, 0.7504, 0.4907]])

# anchor_boxes_kmeaned['coco'] = np.asarray( [[0.0000, 0.0000, 0.0260, 0.0750],
#                                             [0.0000, 0.0000, 0.0161, 0.0269],
#                                             [0.0000, 0.0000, 0.0547, 0.0335],
#                                             [0.0000, 0.0000, 0.0520, 0.1613],
#                                             [0.0000, 0.0000, 0.0657, 0.0783],
#                                             [0.0000, 0.0000, 0.1458, 0.0656],
#                                             [0.0000, 0.0000, 0.1088, 0.3015],
#                                             [0.0000, 0.0000, 0.1285, 0.1558],
#                                             [0.0000, 0.0000, 0.2564, 0.1516],
#                                             [0.0000, 0.0000, 0.1795, 0.5149],
#                                             [0.0000, 0.0000, 0.3053, 0.3588],
#                                             [0.0000, 0.0000, 0.5193, 0.2144],
#                                             [0.0000, 0.0000, 0.3859, 0.7187],
#                                             [0.0000, 0.0000, 0.8255, 0.8502],
#                                             [0.0000, 0.0000, 0.7242, 0.4552]])

feature_sizes = dict()
feature_sizes['600'] = [75, 38, 19, 10, 5]

class anchorBox(object):
    """
    Compute anchorbox coordinates in center-offset form for each source
    feature map.
    
    """
    def __init__(self, input_dim=600, dataset = 'coco', default_ratios= 3):
        super(anchorBox, self).__init__()
        # self.type = cfg.name
        self.image_size = input_dim
        self.anchor_set = dataset
        self.ar = default_ratios
        self.base_set = anchor_boxes_kmeaned[dataset]
        self.feature_size = feature_sizes[str(input_dim)]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_size):
            for i, j in product(range(f), repeat=2):
                f_k = 1 / f
                # unit center x,y
                cx = (j + 0.5) * f_k
                cy = (i + 0.5) * f_k 
                for r in range(self.ar):  
                    anchors.append([cx, cy, self.base_set[k*3+r,2], self.base_set[k*3+r,3]])

        output = torch.FloatTensor(anchors).view(-1, 4)
        # output.clamp_(max=1, min=0)
        return output
