
"""
    Author: Gurkirt Singh
    Purpose: Check number of anchor boxes
    Please don't remove above credits and 
    give star to this repo if it has been useful to you

    Licensed under The MIT License [see LICENSE for details]
    
"""

import math

anchor_areas = [75, 38, 19, 10, 5]  # p3 -> p7
aspect_ratios = [1/2., 1/1., 2/1.]
scale_ratios = [1., 1.5]
anchor_wh = []

for sa in anchor_areas:
    s = 3.05*600/sa
    s *= s
    for ar in aspect_ratios:  # w/h = ar
        h = math.sqrt(s/ar)
        w = ar * h
        for sr in scale_ratios:  # scale
            anchor_h = h*sr
            anchor_w = w*sr
            anchor_wh.append([anchor_w, anchor_h])
num_fms = len(anchor_areas)
print(anchor_wh)

#return torch.Tensor(anchor_wh).view(num_fms, -1, 2)