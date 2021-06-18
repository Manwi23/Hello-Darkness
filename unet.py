import torch
from torch import nn
import torch.nn.functional as F

class Conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_layer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
    
    def forward(self, X):
        return self.layer(X)

    def loss(self, Out, Targets):
        return F.cross_entropy(Out, Targets)


class Down_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down_layer, self).__init__()
        self.pool_down = nn.MaxPool2d(2)
        self.conv = Conv_layer(in_channels, out_channels)

    def forward(self, X):
        out = self.pool_down(X)
        out = self.conv(out)
        return out

    def loss(self, Out, Targets):
        return F.cross_entropy(Out, Targets)


class Up_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up_layer, self).__init__()
        self.pool_up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = Conv_layer(in_channels, out_channels)

    def forward(self, X_d, X_u):
        X_d= self.pool_up(X_d)
        concat_out = torch.cat([X_u, X_d], 1)
        out = self.conv(concat_out)
        return out

    def loss(self, Out, Targets):
        return F.cross_entropy(Out, Targets)


class U_net(nn.Module):
    def __init__(self):
        super(U_net, self).__init__()
        self.down_layer1 = Conv_layer(4, 32)
        self.down_layer2 = Down_layer(32, 64)
        self.down_layer3 = Down_layer(64, 128)
        self.down_layer4 = Down_layer(128, 256)
        self.down_layer5 = Down_layer(256, 512)

        self.up_layer1 = Up_layer(512, 256)
        self.up_layer2 = Up_layer(256, 128)
        self.up_layer3 = Up_layer(128, 64)
        self.up_layer4 = Up_layer(64, 32)
        self.up_layer5 = nn.Conv2d(32, 12, 1)
        self.out_layer = nn.PixelShuffle(2)

    def forward(self, X):
        d1 = self.down_layer1(X)
        d2 = self.down_layer2(d1)
        d3 = self.down_layer3(d2)
        d4 = self.down_layer4(d3)
        d5 = self.down_layer5(d4)

        u1 = self.up_layer1(d5, d4)
        u2 = self.up_layer2(u1, d3)
        u3 = self.up_layer3(u2, d2)
        u4 = self.up_layer4(u3, d1)
        u5 = self.up_layer5(u4)

        out = self.out_layer(u5)
        return out

    def loss(self, Out, Targets):
        return F.cross_entropy(Out, Targets)