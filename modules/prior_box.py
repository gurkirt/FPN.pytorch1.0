import torch
from math import sqrt as sqrt
from itertools import product as product

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, input_dim=300, feature_size = [75, 38, 19, 10, 5],
                                       aspect_ratios =[1 / 2., 1 / 1., 2 / 1.],
                                       scale_ratios = [1., 1.35, 1.75],
                                        is_cuda=True, default_scale= 3): #[2.4, 2.8, 3, 3.2, 3.4]):
        super(PriorBox, self).__init__()
        # self.type = cfg.name
        self.is_cuda = is_cuda
        self.image_size = input_dim
        # number of priors for feature map location (either 4 or 6)
        self.anchor_boxes = len(aspect_ratios)*len(scale_ratios)
        self.num_priors = 0
        self.variance = [0.1, 0.2]
        self.feature_maps = feature_size
        self.scale_ratios = scale_ratios
        self.aspect_ratios = aspect_ratios
        self.default_scale = default_scale
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = 1 / f
                # unit center x,y
                cx = (j + 0.5) * f_k
                cy = (i + 0.5) * f_k
                s = self.default_scale*self.image_size/f
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
