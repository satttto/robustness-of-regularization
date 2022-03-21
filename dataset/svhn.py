import torchvision
import torchvision.transforms as transforms
from .base import Dataset

class SVHN(Dataset):
    
    def fetch(self):
        # fetch
        # the range of each pixel is [0, 1]
        # see https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html?highlight=cifar
        self._trainset = torchvision.datasets.SVHN(root='./data', split='train',  transform=self._train_transform, download=True)
        self._testset  = torchvision.datasets.SVHN(root='./data', split='test', transform=self._test_transform, download=True)

    @property
    def num_classes(self):
        if not hasattr(self, '_trainset'):
            raise NameError('You do not have trainset/testset yet. Call fetch() first')
        return 10