from torch import nn


class ResBlock(nn.Module):
    def __init__(self, C1):
        super(ResBlock, self).__init__()

        self.main_path = nn.Sequential(
                            nn.Conv2d(C1, C1, 3, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(C1, C1, 3, padding=1),
                            nn.ReLU()
                        )

    def forward(self, x):
        y = self.main_path(x)
        return y + x


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            ResBlock(32),
            ResBlock(32),
            ResBlock(32),
            ResBlock(32),
            ResBlock(32),
            nn.Conv2d(32, 12, 1),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.layers(x)