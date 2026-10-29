import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, num_channels, stride=1, use_1x1conv=False):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels,kernel_size=3, padding=1, stride=stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels,kernel_size=3, padding=1, stride=1),      # 第二个卷积不降采样
        )
        self.relu = nn.ReLU(inplace=True)
        self.use_1x1conv = use_1x1conv or (stride != 1)
        if self.use_1x1conv:
            self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels,
                                   kernel_size=1, padding=0, stride=stride, bias=False)
        else:
            self.conv2 = None

def forward(self, x):
        y = self.conv2(x) if self.use_1x1conv else x
        return self.relu(y + self.block(x))


class BaselineNet(nn.Module):
    def __init__(self, num_channels=64, stride=1):
        super(BaselineNet, self).__init__()
        self.conv_in = nn.Conv2d(3, num_channels, 3, padding=1, stride=stride)
        self.relu = nn.ReLU(inplace=True)

        self.block1 = ResBlock(num_channels)
        self.block2 = ResBlock(num_channels)
        self.block3 = ResBlock(num_channels)
        self.block4 = ResBlock(num_channels)

        self.conv_out = nn.Conv2d(num_channels, 3, 1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv_out(x)
        return x

#有BatchNorm的版本
# import torch.nn as nn
#
#
# class ResBlock(nn.Module):
#     def __init__(self, num_channels, stride=1, use_1x1conv=False):
#         super(ResBlock, self).__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels=num_channels, out_channels=num_channels,
#                       kernel_size=3, padding=1, stride=stride),
#             nn.BatchNorm2d(num_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=num_channels, out_channels=num_channels,
#                       kernel_size=3, padding=1, stride=1),      # 第二个卷积不降采样
#             nn.BatchNorm2d(num_channels),
#         )
#         self.relu = nn.ReLU(inplace=True)
#
#         # 通道不变、stride=1 时不需要投影，直接恒等
#         self.use_1x1conv = use_1x1conv or (stride != 1)
#         if self.use_1x1conv:
#             self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels,
#                                    kernel_size=1, padding=0, stride=stride, bias=False)
#             self.bn = nn.BatchNorm2d(num_channels)              # 顺序：conv2 → bn
#         else:
#             self.conv2 = None
#             self.bn = None
#
#     def forward(self, x):
#         y = self.bn(self.conv2(x)) if self.use_1x1conv else x
#         return self.relu(y + self.block(x))
#
#
# class BaselineNet(nn.Module):
#     def __init__(self, num_channels=64, stride=1):
#         super(BaselineNet, self).__init__()
#         self.conv_in = nn.Conv2d(3, num_channels, 3, padding=1, stride=stride)
#         self.bn = nn.BatchNorm2d(num_channels)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.block1 = ResBlock(num_channels)
#         self.block2 = ResBlock(num_channels)
#         self.block3 = ResBlock(num_channels)
#         self.block4 = ResBlock(num_channels)
#
#         self.conv_out = nn.Conv2d(num_channels, 3, 1)
#
#     def forward(self, x):
#         x = self.conv_in(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)
#         x = self.conv_out(x)
#         return x

