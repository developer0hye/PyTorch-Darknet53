import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
from torchsummary import summary


def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())


# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels/2)

        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out


class Darknet53(nn.Module):
    def __init__(self, block, num_classes):
        super(Darknet53, self).__init__()

        self.num_classes = num_classes

        self.layer1 = conv_batch(3, 32)
        self.layer2 = conv_batch(32, 64, stride=2)
        self.layer3_residual_block_1 = self.make_layer(block, in_channels=64, num_blocks=1)
        self.layer4 = conv_batch(64, 128, stride=2)
        self.layer5_residual_block_2 = self.make_layer(block, in_channels=128, num_blocks=2)
        self.layer6 = conv_batch(128, 256, stride=2)
        self.layer7_residual_block_3 = self.make_layer(block, in_channels=256, num_blocks=8)
        self.layer8 = conv_batch(256, 512, stride=2)
        self.layer9_residual_block_4 = self.make_layer(block, in_channels=512, num_blocks=8)
        self.layer10 = conv_batch(512, 1024, stride=2)
        self.layer11_residual_block_5 = self.make_layer(block, in_channels=1024, num_blocks=4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3_residual_block_1(out)
        out = self.layer4(out)
        out = self.layer5_residual_block_2(out)
        out = self.layer6(out)
        out = self.layer7_residual_block_3(out)
        out = self.layer8(out)
        out = self.layer9_residual_block_4(out)
        out = self.layer10(out)
        out = self.layer11_residual_block_5(out)
        out = self.global_avg_pool(out)
        out = self.classifier(out)
        out = out.view(-1, self.num_classes)

        return out

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)


def darknet53(num_classes):
    return Darknet53(DarkResidualBlock, num_classes)

