import torch.nn as nn
class  BaselineNet(nn.Module):
    def __init__(self,in_channels =3,num_channels =64,use_1x1conv = True,stride = 1):
        super(BaselineNet, self).__init__()
        self.all_conv2 =nn.ModuleList()
        self.all_bn2 = nn.ModuleList()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=num_channels,kernel_size=3,padding= 1,stride = stride)
        for i in range(7):
            self.all_conv2.append(nn.Conv2d(in_channels=num_channels,out_channels=num_channels,kernel_size=3,padding= 1,stride = stride))
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=1, padding=0,stride=stride)
        #1×1 卷积永远 padding=0
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(in_channels)
        for i in range(7):
            self.all_bn2.append( nn.BatchNorm2d(num_channels))
        self.relu= nn.ReLU(inplace = True)
        self.conv_out = nn.Conv2d(num_channels, in_channels, kernel_size=3, padding=1)

    def forward(self,X):
        Y = self.relu(self.conv1(self.bn1(X)))
        for i in range(7):
            Y =self.relu(self.all_conv2[i](self.all_bn2[i](Y)))
        if self.conv3:
            shortcut = self.conv3(X)# 不能加bn1，这个一定不能缩放，要保持干净
        Y =Y + shortcut
        Y = self.relu(Y)
        rain = self.conv_out(Y)
        return X - rain


