"""GroupNorm convolutional networks used for FMNIST, CIFAR-10 and Tiny-ImageNet."""

import torch.nn as nn


def _group_norm(channels: int, groups: int = 8) -> nn.GroupNorm:
    g = groups
    while g > 1 and channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


class ConvNetGN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, widths, double_conv: bool = False):
        super().__init__()
        layers, c = [], in_channels
        for w in widths:
            layers += [nn.Conv2d(c, w, 3, padding=1, bias=False), _group_norm(w), nn.ReLU(inplace=True)]
            if double_conv:
                layers += [nn.Conv2d(w, w, 3, padding=1, bias=False), _group_norm(w), nn.ReLU(inplace=True)]
            layers += [nn.MaxPool2d(2, 2)]
            c = w
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(c, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def build_model(dataset: str) -> nn.Module:
    if dataset == "fashionmnist":
        return ConvNetGN(1, 10, (32, 64, 128))
    if dataset == "cifar10":
        return ConvNetGN(3, 10, (32, 64, 128), double_conv=True)
    if dataset == "tinyimagenet":
        return ConvNetGN(3, 200, (32, 64, 128, 256))
    raise ValueError(f"unsupported dataset: {dataset}")
