import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F

from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import build_resnet_backbone, FPN
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY

import math
import warnings

import torch
from torch import nn

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

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
    
class Yolov8Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        d = 0.33 #depth_multiple for yolov8n

        self.conv_1 = Conv(3, 64, k=3, s=2)
        self.conv_2 = Conv(64, 128, k=3, s=2)
        self.c2f_1 = C2f(128, 128, shortcut=True, n=3*d)
        self.conv_3 = Conv(128,256, k=3, s=2)
        self.c2f_2 = C2f(256, 256, shortcut=True, n=6*d)
        self.conv_4 = Conv(256, 512, k=3, s=2)
        self.c2f_3 = C2f(512, 512, shortcut=True, n=6*d)
        self.conv_5 = Conv(512, 1024, k=3, s=2)
        self.c2f_4 = C2f(1024, 1024, shortcut=True, n=3*d)
        self.sppf = SPPF(1024, 1024)
    
    def forward(self, x):
        out_conv_1 = self.conv_1(x)
        out_conv_2 = self.conv_2(out_conv_1)
        out_c2f_1 = self.c2f_1(out_conv_2)
        out_conv_3 = self.conv_3(out_c2f_1)

        p3 = self.c2f_2(out_conv_3)

        out_conv_4 = self.conv_4(p3)

        p4 = self.c2f_3(out_conv_4)

        out_conv_5 = self.conv_5(p4)
        out_c2f_4 = self.c2f_4(out_conv_5)

        p5 = self.sppf(out_c2f_4)

        return [p3, p4, p5]

@BACKBONE_REGISTRY.register()
def build_yolov8_backbone():
    return Yolov8Backbone()