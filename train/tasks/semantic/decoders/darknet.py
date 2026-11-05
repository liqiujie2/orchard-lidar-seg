
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, bn_d=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0], momentum=bn_d)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1], momentum=bn_d)
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out += residual
        return out

class Decoder(nn.Module):

    def __init__(self, params, stub_skips, OS=8, feature_depth=256):
        super(Decoder, self).__init__()
        self.backbone_OS = OS
        self.backbone_feature_depth = feature_depth
        self.drop_prob = params["dropout"]
        self.bn_d = params["bn_d"]

        # stride play
        self.strides = [2, 2, 2]
        # check current stride
        current_os = 1
        for s in self.strides:
            current_os *= s
        print("Decoder original OS: ", int(current_os))
        # redo strides according to needed stride
        for i, stride in enumerate(self.strides):
            if int(current_os) != self.backbone_OS:
                if stride == 2:
                    current_os /= 2
                    self.strides[i] = 1
                if int(current_os) == self.backbone_OS:
                    break
        print("Decoder new OS: ", int(current_os))
        print("Decoder strides: ", self.strides)

        self.dec3 = self._make_dec_layer(BasicBlock, [self.backbone_feature_depth, 128], bn_d=self.bn_d,
                                         stride=self.strides[0])
        self.dec2 = self._make_dec_layer(BasicBlock, [128, 64], bn_d=self.bn_d, stride=self.strides[1])
        self.dec1 = self._make_dec_layer(BasicBlock, [64, 32], bn_d=self.bn_d, stride=self.strides[2])

        self.layers = [self.dec3, self.dec2, self.dec1]

        self.dropout = nn.Dropout2d(self.drop_prob)

        self.last_channels = 32

    def _make_dec_layer(self, block, planes, bn_d=0.1, stride=2):
        layers = []

        #  downsample
        if stride == 2:
            layers.append(("upconv", nn.ConvTranspose2d(planes[0], planes[1],
                                                        kernel_size=[1, 4], stride=[1, 2],
                                                        padding=[0, 1])))
        else:
            layers.append(("conv", nn.Conv2d(planes[0], planes[1],
                                             kernel_size=3, padding=1)))
        layers.append(("bn", nn.BatchNorm2d(planes[1], momentum=bn_d)))
        layers.append(("relu", nn.LeakyReLU(0.1)))

        #  blocks
        layers.append(("residual", block(planes[1], planes, bn_d)))

        return nn.Sequential(OrderedDict(layers))

    def run_layer(self, x, layer, skips, os):
        feats = layer(x)
        if feats.shape[-1] > x.shape[-1]:
            os = os // 2
            skip_feat = skips[os]
            skip_w = skip_feat.shape[-1]
            curr_w = feats.shape[-1]

            if curr_w < skip_w:
                diff = skip_w - curr_w
                feats = F.pad(feats, pad=(0, diff, 0, 0), mode="constant", value=0.0)
            elif curr_w > skip_w:

                feats = feats[:, :, :, :skip_w]

            feats = feats + skip_feat
        x = feats
        return x, skips, os

    def forward(self, x, skips):
        os = self.backbone_OS

        x, skips, os = self.run_layer(x, self.dec3, skips, os)
        x, skips, os = self.run_layer(x, self.dec2, skips, os)
        x, skips, os = self.run_layer(x, self.dec1, skips, os)

        x = self.dropout(x)
        return x

    def get_last_depth(self):
        return self.last_channels
