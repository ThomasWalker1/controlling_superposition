import torch
import torch.nn as nn

class ReLUFunction(nn.Module):
    def __init__(self, negative_slope=0.0, inplace=True):
        super(ReLUFunction, self).__init__()
        self.negative_slope=negative_slope
        self.inplace=inplace
        
    def forward(self, x):
        if self.inplace:
            return x.clamp_(min=0)-self.negative_slope*(-x).clamp_(min=0)
        else:
            return torch.clamp(x,min=0)-self.negative_slope*torch.camp(-x,min=0)
        
class AlexNet(nn.Module):
    def __init__(self, output_dim, negative_slope):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),  # in_channels, out_channels, kernel_size, stride, padding
            nn.MaxPool2d(2),  # kernel_size
            ReLUFunction(negative_slope),
            nn.Conv2d(64, 192, 3, padding=1),
            nn.MaxPool2d(2),
            ReLUFunction(negative_slope),
            nn.Conv2d(192, 384, 3, padding=1),
            ReLUFunction(negative_slope),
            nn.Conv2d(384, 256, 3, padding=1),
            ReLUFunction(negative_slope),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.MaxPool2d(2),
            ReLUFunction(negative_slope)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h