import torch
import torch.nn as nn

class VGG13(nn.Module):
    def __init__(self, num_classes=10, init_weights=True, dropout=0.5):
        super(VGG13, self).__init__()
        self.layer1 = self._make_layer(3, 64)
        self.layer2 = self._make_layer(64, 128)
        self.layer3 = self._make_layer(128, 256)
        self.layer4 = self._make_layer(256, 512)
        self.layer5 = self._make_layer(512, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
    
    def _make_layer(in_channels, dim):
        conv2d_1 = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
        conv2d_2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        layers = [
            conv2d_1,
            nn.ReLU(inplace=True),
            conv2d_2,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def vgg13(**kwargs):
    return VGG13(**kwargs)
    


    

