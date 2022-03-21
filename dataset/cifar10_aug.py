import torch
import torchvision
import torchvision.transforms as transforms
from .cifar10 import CIFAR10

class CIFAR10Aug(CIFAR10):
    def __init__(self, batch_size=128, methods=[]):
        '''
        Augment the data based on the official Pytorch Lightning article below
        https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/cifar10-baseline.html

        random erasing: https://ojs.aaai.org/index.php/AAAI/article/download/7000/6854
        '''
        transform_list = [transforms.ToTensor()]
        if 'crop' in methods:
            print('crop')
            transform_list.append(transforms.RandomCrop(32, padding=4))
        if 'hflip' in methods:
            print('hflip')
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        if 'rotate' in methods:
            print('rotate')
            transform_list.append(transforms.RandomRotation(degrees=20))
        if 'erase' in methods:
            print('erase')
            transform_list.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3))) 
        self._train_transform = transforms.Compose(transform_list)
        self._test_transform = transforms.Compose([transforms.ToTensor()])
        self._batch_size = batch_size