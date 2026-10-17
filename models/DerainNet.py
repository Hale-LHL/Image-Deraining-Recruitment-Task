import torch
from torch import nn


class DerainNet(nn.Module):
    def __init__(self,c =64):
        super(DerainNet, self).__init__()
        self.all_conv_1 = nn.ModuleList()
        self.all_conv_2 = nn.ModuleList()
        self.all_conv_4 = nn.ModuleList()
        self.all_bn_1 = nn.ModuleList()
        self.all_bn_2 = nn.ModuleList()
        self.all_bn_4 = nn.ModuleList()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=c,kernel_size=3,padding=1)
        self.conv_out = nn.Conv2d(in_channels=3*c, out_channels=3, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for _ in range(8):
            self.all_conv_1.append(nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1))
        for _ in range(8):
            self.all_conv_2.append(nn.Conv2d(in_channels=c,out_channels=c,kernel_size=3,padding=1))
        for _ in range(8):
            self.all_conv_4.append(nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1))
        for _ in range(8):
            self.all_bn_1.append(nn.BatchNorm2d(c))
        for _ in range(8):
            self.all_bn_2.append(nn.BatchNorm2d(c))
        for _ in range(8):
            self.all_bn_4.append(nn.BatchNorm2d(c))
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.ReLU(inplace = True)


    def forward(self,x):
        x_1 = self.conv1(x)
        x_2 = self.pool(x_1)
        x_4 = self.pool(x_2)
        y_1,y_2,y_4=x_1,x_2,x_4
        for i in range(8):
            y_1 = self.relu(self.all_conv_1[i](self.all_bn_1[i](y_1)))
        y_1 = x_1 +y_1
        for i in range(8):
            y_2 = self.relu(self.all_conv_2[i](self.all_bn_2[i](y_2)))
        y_2 = x_2 + y_2
        for i in range(8):
            y_4 = self.relu(self.all_conv_4[i](self.all_bn_4[i](y_4)))
        y_4 = x_4 + y_4

        y_2_up = self.upsample(y_2)
        y_4_up = self.upsample(self.upsample(y_4))

        residual =torch.concat((y_1,y_2_up,y_4_up),dim = 1)
        residual_3 = self.conv_out(residual)
        out = x -residual_3
        return out




