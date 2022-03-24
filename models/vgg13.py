'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.autograd import Variable



class VGG13(nn.Module):
    def __init__(self):
        super(VGG13, self).__init__()
        self.layer1 = self._make_layer(3, 64)
        self.layer2 = self._make_layer(64, 128)
        self.layer3 = self._make_layer(128, 256)
        self.layer4 = self._make_layer(256, 512)
        self.layer5 = self._make_layer(512, 512)
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layer(self, in_channels, out_channels):
        layers = []
        layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                       nn.BatchNorm2d(out_channels),
                       nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                       nn.BatchNorm2d(out_channels),
                       nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)


def vgg13(**kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return VGG13()

