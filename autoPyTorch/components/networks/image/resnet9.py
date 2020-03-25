# Code from https://github.com/wbaek/torchskeleton/releases/tag/v0.2.1_dawnbench_cifar10_release
import torch
import torch.nn as nn

from autoPyTorch.components.networks.base_net import BaseImageNet

class Mul(torch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1, bn=True, activation=True):
    op = [
            torch.nn.Conv2d(channels_in, channels_out,
                            kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
    ]
    if bn:
        op.append(torch.nn.BatchNorm2d(channels_out))
    if activation:
        op.append(torch.nn.ReLU(inplace=True))
    return torch.nn.Sequential(*op)


class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


def build_network(num_class=10):
    return torch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        # torch.nn.MaxPool2d(2),

        Residual(torch.nn.Sequential(
            conv_bn(128, 128),
            conv_bn(128, 128),
        )),

        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),

        Residual(torch.nn.Sequential(
            conv_bn(256, 256),
            conv_bn(256, 256),
        )),

        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),

        torch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        torch.nn.Linear(128, num_class, bias=False),
        Mul(0.2)
    )


class ResNet9(BaseImageNet):
    def __init__(self, config, in_features, out_features, final_activation, **kwargs):
        super(ResNet9, self).__init__(config, in_features, out_features, final_activation)
        self.layers = build_network(num_class=out_features)

    def forward(self, x):
        x = self.layers.forward(x)
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        return x
