import os

import numpy as np
import torch
import torch.nn as nn
import netron
from torchviz import make_dot
import torch.onnx


class ResBlock(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()

        self.conv0 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, stride=1)
        self.norm0 = nn.BatchNorm2d(num_channels)
        self.act = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, stride=1)
        self.norm1 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        out = self.conv0(x)
        out = self.norm0(out)
        out = self.act(out)
        out = self.conv1(out)
        # out = self.norm1(out)

        return x + out  # self.act(x + out)


class BottleneckBlock(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.act = nn.GELU()#nn.LeakyReLU(0.2, inplace=True)
        self.n = 4
        self.conv0 = nn.Conv2d(num_channels, num_channels // self.n, kernel_size=1, padding=0)
        self.norm0 = nn.BatchNorm2d(num_channels // self.n)
        self.conv1 = nn.Conv2d(num_channels // self.n, num_channels // self.n, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(num_channels // self.n)
        self.conv2 = nn.Conv2d(num_channels // self.n, num_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.conv0(x)
        out = self.norm0(out)
        out = self.act(out)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)

        return self.act(x + out)


class ResTruck(nn.Module):
    def __init__(self,
                 num_channels: int,
                 num_blocks: int,
                 block_type: str = 'classic'):
        super().__init__()

        truck = []
        for i in range(num_blocks):
            if block_type == 'classic':
                truck += [ResBlock(num_channels)]
            elif block_type == 'bottleneck':
                truck += [BottleneckBlock(num_channels)]
            else:
                raise NotImplementedError(f'{block_type} is not implemented')
        self.truck = nn.Sequential(*truck)

    def forward(self, x):
        return self.truck(x)


class PsevdoResNet_2(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_channels: int = 128,
                 out_channels: int = 33,
                 block_type: str = 'bottleneck',
                 stride: int = 1,
                 padding: int = 1):
        super().__init__()

        self.conv0 = nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=stride)
        # self.norm
        self.act = nn.GELU()#nn.LeakyReLU(0.2, inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.layer1 = ResTruck(num_channels, 3, block_type=block_type)
        self.conv1 = nn.Conv2d(num_channels, 2 * num_channels, 3, padding=padding, stride=stride)
        self.layer2 = ResTruck(2 * num_channels, 4, block_type=block_type)
        self.conv2 = nn.Conv2d(2 * num_channels, 4 * num_channels, 3, padding=padding, stride=stride)
        self.layer3 = ResTruck(4 * num_channels, 6, block_type=block_type)
        self.conv3 = nn.Conv2d(4 * num_channels, 4 * num_channels, 3, padding=padding, stride=stride)
        self.layer4 = ResTruck(4 * num_channels, 3, block_type=block_type)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(4 * num_channels, out_channels)

    def forward(self, x):
        out = self.conv0(x)
        out = self.act(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.conv1(out)
        out = self.layer2(out)
        out = self.conv2(out)
        out = self.layer3(out)
        out = self.conv3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.linear(out)

        return out

# if __name__ == '__main__':
#     from utils.visualization_model_arch import Convert_ONNX
#
#     # Let's build our model
#     # train(5)
#     # print('Finished Training')
#
#     # Test which classes performed well
#     # testAccuracy()
#
#     # Let's load the model we just created and test the accuracy per label
#     model = PsevdoResNet().eval()
#     path = "/home/rain/vs_code/relize/saves_weights/ResNet/ResNet_in_ch_1_num_ch_128_out_ch_33_padding_1_stride_1_blk_bottleneck_lr_st_opt_name_AdamW_eps_1e-8_lr_0.001_bts1_0.9_bts2_0.999_w_dc_0.001_schr_g_0.6_amp_True_bench_True_detrm_False_ep_10.pth"
#     model.load_state_dict(torch.load(path))
#
#     # Test with batch of images
#     # testBatch()
#     # Test how the classes performed
#     # testClassess()
#
#     # Conversion to ONNX
#     Convert_ONNX(model, (33, 128, 1, 32))
#     # Convert_ONNX(model,(32,1,128,33))
