from torch import nn
import torch
from .resnet18 import BasicBlock, Bottleneck, ResNet18
from utils import DropBlock2D, LinearScheduler

class ResNetDropBlock(ResNet18):
    def __init__(self, block, layers, num_classes=10, drop_prob=0., block_size=5, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, **kwargs):
        super().__init__(block, layers, num_classes, zero_init_residual,
                        groups, width_per_group, replace_stride_with_dilation,
                        norm_layer, **kwargs)
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=drop_prob, block_size=block_size),
            start_value=0.,
            stop_value=drop_prob,
            nr_steps=110 # the number of epochs
        )

    def forward(self, x):
        self.dropblock.step()  # increment number of iterations

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.dropblock(self.layer1(x))
        x = self.dropblock(self.layer2(x))
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet18_dropblock(**kwargs):
    return ResNetDropBlock(BasicBlock, [2, 2, 2, 2], **kwargs)