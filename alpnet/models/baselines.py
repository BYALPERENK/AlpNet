"""Baselines B1-B4 for 1x64x64 inputs.

B1 LeNet     - weak reference: two 5x5 convs, no norm, no residual, flatten head
B2 PlainCNN  - medium: VGG-style, 4 stages x 2 (Conv3x3-BN-ReLU), GAP head
B3 ResNet18  - strong: torchvision ResNet-18 with a small-image stem (3x3, stride 1, no maxpool)
B4 MobileViTv2-1.75 - modern CNN/transformer hybrid (timm), stem stride 1
"""

import timm
import torch.nn as nn
from torchvision.models import resnet18


class LeNet(nn.Module):
    def __init__(self, num_classes: int = 15):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5), nn.ReLU(), nn.MaxPool2d(2),    # 64 -> 60 -> 30
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),   # 30 -> 26 -> 13
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 13 * 13, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def _conv_bn_relu(cin: int, cout: int) -> list[nn.Module]:
    return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class PlainCNN(nn.Module):
    def __init__(self, num_classes: int = 15, widths=(32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        layers, cin = [], 1
        for i, w in enumerate(widths):
            if i > 0:
                layers.append(nn.MaxPool2d(2))                 # 64 -> 32 -> 16 -> 8
            layers += _conv_bn_relu(cin, w) + _conv_bn_relu(w, w)
            cin = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(cin, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def resnet18_64(num_classes: int = 15) -> nn.Module:
    m = resnet18(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False)  # was 7x7 stride 2 (3 channels)
    m.maxpool = nn.Identity()                                       # feature maps 64 -> 32 -> 16 -> 8
    nn.init.kaiming_normal_(m.conv1.weight, mode="fan_out", nonlinearity="relu")
    return m


def mobilevitv2_175_64(num_classes: int = 15) -> nn.Module:
    m = timm.create_model("mobilevitv2_175", pretrained=False, in_chans=1, num_classes=num_classes)
    m.stem.conv.stride = (1, 1)  # was 2: keeps the attention stages at 16/8/4 px instead of 8/4/2
    return m


MODELS = {
    "b1_lenet": LeNet,
    "b2_plain": PlainCNN,
    "b3_resnet18": resnet18_64,
    "b4_mobilevitv2": mobilevitv2_175_64,
}
