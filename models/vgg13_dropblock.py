from .vgg13 import VGG13
import torch
from utils import DropBlock2D, LinearScheduler

class VGG13DropBlock(VGG13):
    def __init__(self, num_classes=10, init_weights=True, dropout=0.5, drop_prob=0, block_size=5, **kwargs):
        super().__init__(num_classes, init_weights, dropout, **kwargs)
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=drop_prob, block_size=block_size),
            start_value=0,
            stop_value=drop_prob,
            nr_steps=110
        )

    def forward(self, x):
        self.dropblock.step()

        x = self.layer1(x)
        x = self.dropblock(self.layer2(x))
        x = self.dropblock(self.layer3(x))
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    
def vgg13_dropblock(**kwargs):
    return VGG13DropBlock(**kwargs)
