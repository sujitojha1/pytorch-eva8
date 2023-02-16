# Import the necessary modules
import torch
import torch.nn as nn

# Define the ResBlock class
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        self.resConv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

    def forward(self, x):
        out = self.resConv(x)
        out += self.shortcut(x)
        return out


class CustomResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()

        self.prep_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer1 = self._make_layer(64, 128, stride=1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3,
                               stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer3 = self._make_layer(256, 512, stride=1)

        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self,in_channels, out_channels, num_blocks, stride):

        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.prep_layer(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out