import torch
from math import sqrt as sqrt
from itertools import product as product
import numpy as np

class anchorBox(object):
    """Compute anchorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, anchor_type, input_dim=600, feature_size = [75, 38, 19, 10, 5],
                                       aspect_ratios =[0.5, 1 / 1., 1.5],
                                       scale_ratios = [1.,1.34, 1.67],
                                       dataset='all'):
        super(anchorBox, self).__init__()
        # self.type = cfg.name
        self.image_size = input_dim
        self.num_anchors = 0
        self.variance = [0.1, 0.2]
        self.feature_maps = feature_size
        self.aspect_ratios = aspect_ratios
        self.scale_ratios = scale_ratios
        self.default_scale= [2.4, 2.8, 3, 3.2, 3.4]
        if anchor_type not in ['pdf9']: # gives 3 anchors per scale # default ps pre-defined 9
            self.aspect_ratios = [0.54, 1 / 1., 1.45]
            self.scale_ratios = [1.0]
            self.default_scale= [2.8, 3, 3, 3.2, 3.4]
        # self.default_scale = 2.8
        
        self.anchor_boxes = len(self.aspect_ratios)*len(self.scale_ratios)
        self.ar = self.anchor_boxes
        print(self.scale_ratios, anchor_type, self.ar)

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = 1 / f
                # unit center x,y
                cx = (j + 0.5) * f_k
                cy = (i + 0.5) * f_k
                s = self.default_scale[k]*self.image_size/f
                s *= s
                for ar in self.aspect_ratios:  # w/h = ar
                    h = sqrt(s / ar)
                    w = ar * h
                    for sr in self.scale_ratios:  # scale
                        anchor_h = h * sr
                        anchor_w = w * sr
                        anchors.append([cx, cy, anchor_w/self.image_size, anchor_h/self.image_size])

        output = torch.FloatTensor(anchors).view(-1, 4)
        # output.clamp_(max=1, min=0)
        return output
