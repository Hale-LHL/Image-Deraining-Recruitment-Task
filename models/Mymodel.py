from torch import nn


class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3,padding=1 )
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(64)
        self.bn8 = nn.BatchNorm2d(64)

    def forward(self, x):
        y = self.relu(self.conv1(self.bn1(x)))
        y = self.relu(self.conv2(self.bn2(y)))
        y = self.relu(self.conv3(self.bn3(y)))
        y = self.relu(self.conv4(self.bn4(y)))
        y = self.relu(self.conv5(self.bn5(y)))
        y = self.relu(self.conv6(self.bn6(y)))
        y = self.relu(self.conv7(self.bn7(y)))
        y = self.conv_out(self.bn8(y))
        return x-y