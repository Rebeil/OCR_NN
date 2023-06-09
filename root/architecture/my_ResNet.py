import torch.nn as nn
#изменить страйд

class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        self.conv0 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.norm0 = nn.BatchNorm2d(num_channels)
        self.act = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        out = self.conv0(x)
        out = self.norm0(out)
        out = self.act(out)
        out = self.conv1(out)
        # out = self.norm1(out)

        return x + out  # self.act(x + out)


class BottleneckBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.act = nn.LeakyReLU(0.2)

        self.conv0 = nn.Conv2d(num_channels, num_channels // 4, kernel_size=1, padding=0)
        self.norm0 = nn.BatchNorm2d(num_channels // 4)
        self.conv1 = nn.Conv2d(num_channels // 4, num_channels // 4, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(num_channels // 4)
        self.conv2 = nn.Conv2d(num_channels // 4, num_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.conv0(x)
        out = self.norm0(out)
        out = self.act(out)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)

        return x + out  # self.act(x + out)


class ResTruck(nn.Module):
    def __init__(self, num_channels, num_blocks, block_type='classic'):
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


class PsevdoResNet(nn.Module):
    def __init__(self,
                 in_channels:int,
                 num_channels:int,
                 out_channels:int,
                 block_type:str='bottleneck'):
        super().__init__()

        self.conv0 = nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=2)
        # self.norm
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.layer1 = ResTruck(num_channels, 3, block_type=block_type)
        self.conv1 = nn.Conv2d(num_channels, 2 * num_channels, 3, padding=1, stride=2)
        self.layer2 = ResTruck(2 * num_channels, 4, block_type=block_type)
        self.conv2 = nn.Conv2d(2 * num_channels, 4 * num_channels, 3, padding=1, stride=2)
        self.layer3 = ResTruck(4 * num_channels, 6, block_type=block_type)
        self.conv3 = nn.Conv2d(4 * num_channels, 4 * num_channels, 3, padding=1, stride=2)
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
