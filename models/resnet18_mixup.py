import random
import numpy as np
import torch
import torch.nn.functional as F
from .resnet18 import BasicBlock, Bottleneck, ResNet18
from  utils import mixup_data

class ResNetManiMixup(ResNet18):
    def forward(self, x, target=None, mixup=False, mixup_alpha=0.1, layer_mix=None, mixup_hidden=False):
        if mixup:
            if layer_mix == None and mixup_hidden == False:
                layer_mix = random.randint(0, 2)
            elif layer_mix == None and mixup_hidden == True:
                layer_mix = random.randint(1, 2)

            if layer_mix == 0:
                x, y_a, y_b, lam = mixup_data(x, target, mixup_alpha)
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)

            if layer_mix == 1:
                x, y_a, y_b, lam = mixup_data(x, target, mixup_alpha)
            x = self.layer2(x)

            if layer_mix == 2:
                x, y_a, y_b, lam = mixup_data(x, target, mixup_alpha)
            x = self.layer3(x)

            if layer_mix == 3:
                x, y_a, y_b, lam = mixup_data(x, target, mixup_alpha)
            x = self.layer4(x)

            if layer_mix == 4:
                x, y_a, y_b, lam = mixup_data(x, target, mixup_alpha)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            
            lam = torch.tensor(lam).cuda()
            lam = lam.item() # a 0-dim tensor to a value

            return x, y_a, y_b, lam

        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        
            return x

def resnet18_mixup(**kwargs):
    return ResNetManiMixup(BasicBlock, [2, 2, 2, 2], **kwargs)