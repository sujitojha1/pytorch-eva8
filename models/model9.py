# Import the necessary modules
import torch
import torch.nn as nn
import torch.nn.functional as F

class ULTIMUS(nn.Module):
    def __init__(self):
        super().__init__()

        self.keyLayer = nn.Linear(48, 8)
        self.QueryLayer = nn.Linear(48,8)
        self.ValueLayer = nn.Linear(48,8)

        self.scale = 8 ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.outputFC = nn.Linear(8,48)

    def forward(self, x):
        K = self.keyLayer(x)
        Q = self.QueryLayer(x)
        V = self.ValueLayer(x)

        dots = torch.matmul(Q.transpose(-1, -2), K) * self.scale
        AM = self.attend(dots)
        Z = torch.matmul(V, AM)

        out = self.outputFC(Z)
        return out


class net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.convBlock = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.ULTIMUS1 = ULTIMUS()
        self.ULTIMUS2 = ULTIMUS()
        self.ULTIMUS3 = ULTIMUS()
        self.ULTIMUS4 = ULTIMUS()

        self.FC = nn.Linear(48,10)

    def forward(self, x):
        out = self.convBlock(x)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.ULTIMUS1(out)
        out = self.ULTIMUS2(out)
        out = self.ULTIMUS3(out)
        out = self.ULTIMUS4(out)

        out = self.FC(out)
        return F.log_softmax(out, dim=-1)


