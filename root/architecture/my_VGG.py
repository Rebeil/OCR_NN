import torch.nn as nn


class VGG16(nn.Module):
    """
    Args:
        out_nc (int): Кол-во выходных классов
    """
    def __init__(self,in_channels:int, num_channels:int, out_channels:int):
        super().__init__()

        self.act = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.conv1_1 = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, stride=1)
        self.conv1_2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, stride=1)

        self.conv2_1 = nn.Conv2d(num_channels, 2*num_channels, kernel_size=3, padding=1, stride=1)
        self.conv2_2 = nn.Conv2d(2*num_channels, 2*num_channels, kernel_size=3, padding=1, stride=1)

        self.conv3_1 = nn.Conv2d(2*num_channels, 2*num_channels, kernel_size=3, padding=1, stride=1)
        self.conv3_2 = nn.Conv2d(2*num_channels, 2*num_channels, kernel_size=3, padding=1, stride=1)
        self.conv3_3 = nn.Conv2d(2*num_channels, 2*num_channels, kernel_size=3, padding=1, stride=1)

        self.conv4_1 = nn.Conv2d(2*num_channels, 2*num_channels, kernel_size=3, padding=1, stride=1)
        self.conv4_2 = nn.Conv2d(2*num_channels, 2*num_channels, kernel_size=3, padding=1, stride=1)
        self.conv4_3 = nn.Conv2d(2*num_channels, 2*num_channels, kernel_size=3, padding=1, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        #         self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        #         self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        #         self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(128, 128)
        # self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(128, out_channels)

    def forward(self, x):
        out = self.conv1_1(x)
        out = self.act(out)
        out = self.conv1_2(out)
        out = self.act(out)

        out = self.maxpool(out)

        out = self.conv2_1(out)
        out = self.act(out)
        out = self.conv2_2(out)
        out = self.act(out)

        out = self.maxpool(out)

        out = self.conv3_1(out)
        out = self.act(out)
        out = self.conv3_2(out)
        out = self.act(out)
        out = self.conv3_3(out)
        out = self.act(out)

        out = self.maxpool(out)

        out = self.conv4_1(out)
        out = self.act(out)
        out = self.conv4_2(out)
        out = self.act(out)
        out = self.conv4_3(out)
        out = self.act(out)

        out = self.maxpool(out)

        #         out = self.conv5_1(out)
        #         out = self.act(out)
        #         out = self.conv5_2(out)
        #         out = self.act(out)
        #         out = self.conv5_3(out)
        #         out = self.act(out)

        #         out = self.maxpool(out)
        out = self.avgpool(out)
        out = self.flat(out)

        out = self.fc1(out)
        out = self.act(out)
        #         out = self.fc2(out)
        #         out = self.act(out)
        out = self.fc3(out)

        return out
