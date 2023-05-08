import math
from typing import Dict, List

import torch
from detectron2.layers import NaiveSyncBatchNorm, ShapeSpec
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY

import torch
from torch import nn
from torch.nn import functional as F
from ubteacher.utils.comm import compute_locations

# from ultralytics.yolo.utils.tal import dist2bbox, make_anchors
#TODO: add this functions

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
class C2f(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class DFL(nn.Module):
    # Integral module of Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

class Detect(nn.Module):
    # YOLOv8 Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    # TODO: ch understanding

    def __init__(self, nc=80, ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

@PROPOSAL_GENERATOR_REGISTRY.register()
class YOLOv8(nn.Module):
    pass

class YOLOv8Head(nn.Module): #? How to declare p3, p4, p5
    def __init__(self, p3, p4, p5):
        super().__init__()
        d = 0.33
        self.p3 = p3
        self.p4 = p4
        self.p5 = p5

        self.ump = nn.Upsample(scale_factor=2)
        self.concat = Concat() #! Concatenate list of tensors
        self.c2f_1 = C2f(512, 512, n=3*d)
        self.c2f_2 = C2f(256, 256, n=3*d)

        self.conv_1 = Conv(256, 256, k=3, s=2)
        self.c2f_3 = C2f(512, 512, n=3*d)
        self.conv_2 = Conv(512, 512, k=3, s=2)
        self.c2f_4 = C2f(1024, 1024, n=3*d)

        self.detect = Detect(nc=20)

    def forward(self):
        out_ump_1 = self.ump(self.p5)
        out_concat_1 = self.concat([out_ump_1, self.p4])
        out_c2f_1 = self.c2f_1(out_concat_1)
        out_ump_2 = self.ump(out_c2f_1)
        out_concat_2 = self.concat([out_ump_2, self.p3])
        out_c2f_2 = self.c2f_2(out_concat_2)

        out_conv_1 = self.conv_1(out_c2f_2)
        out_concat_3 = self.concat([out_c2f_1, out_conv_1])

        out_c2f_3 = self.c2f_3(out_concat_3)
        out_conv_2 = self.conv_2(out_c2f_3)
        out_concat_4 = self.concat([out_conv_2, out_conv_2])
        out_c2f_4 = self.c2f_4(out_concat_4)

        detect_1 = self.detect(out_c2f_2)
        detect_2 = self.detect(out_c2f_3)
        detect_3 = self.detect(out_c2f_4)

        return [detect_1, detect_2, detect_3] #TODO: understand how to convert to proposal head