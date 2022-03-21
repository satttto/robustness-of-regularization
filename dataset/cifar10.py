import torchvision
import torchvision.transforms as transforms
from .base import Dataset

class CIFAR10(Dataset):
    
    def fetch(self):
        # fetch
        # the range of each pixel is [0, 1]
        # see https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html?highlight=cifar
        self._trainset = torchvision.datasets.CIFAR10(root='./data', train=True,  transform=self._train_transform, download=True)
        self._testset  = torchvision.datasets.CIFAR10(root='./data', train=False, transform=self._test_transform, download=True)
