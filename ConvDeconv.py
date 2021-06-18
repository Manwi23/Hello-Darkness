from torch import nn
import torchvision

class ConvDeconv(nn.Module):
    def __init__(self):
        super(ConvDeconv, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(4, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 12, 3),
            nn.PixelShuffle(2)
        )

        self.mse = nn.MSELoss()

    def forward(self, x):
        return self.layers(x)

    def loss(self, x, y):
        if y.shape != x.shape:
            trans = torchvision.transforms.CenterCrop((x.shape[2], x.shape[3]))
            y = trans(y)
        return self.mse(x,y)