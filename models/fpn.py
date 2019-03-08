
""" FPN network Classes

Author: Gurkirt Singh
Inspired from https://github.com/kuangliu/pytorch-retinanet and
https://github.com/gurkirt/realtime-action-detection

"""
from models.base_models import base_models
import torch, math
import torch.nn as nn


class FPN(nn.Module):
    """Feature Pyramid Network Architecture
    The network is composed of a base network followed by the
    added Head conv layers.  Each head layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1612.03144.pdf for more details.

    Args:
        base:
        head: head consists of loc and conf conv layers

    """

    def __init__(self, base, ar, head_size, num_classes):
        super(FPN, self).__init__()

        self.num_classes = num_classes
        # TODO: implement __call__ in PriorBox
        self.ar = ar
        self.base_net = base
        self.loc = self.make_head(head_size, self.ar * 4)
        self.conf = self.make_head(head_size, self.ar * num_classes)
        self.softmax = nn.Softmax().cuda()
        self.sigmoid = nn.Sigmoid().cuda()
        # self.detect = Detect(num_classes, 0, 200, 0.001, 0.45)

    def forward(self, x):

        sources = self.base_net(x)
        loc = list()
        conf = list()

        # apply multibox head to source layers
        # c = 0
        for x in sources:
            # print('LEVEL c ', c, torch.sum(x.data))
            # c+=1
            loc.append(self.loc(x).permute(0, 2, 3, 1).contiguous())
            conf.append(self.conf(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        output = (loc.view(loc.size(0), -1, 4),
                  conf.view(conf.size(0), -1, self.num_classes)
                  )
        return output

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()



    def make_head(self, head_size, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(head_size, head_size, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(head_size, out_planes, kernel_size=3, stride=1, padding=1))
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
                #m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        return layers

def build_fpn(modelname='resnet50', ar=6, head_size = 256, num_classes=80):

    return FPN(base_models(modelname = modelname), ar, head_size, num_classes)
