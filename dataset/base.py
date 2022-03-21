import functools
import torch
import torchvision.transforms as transforms

'''
Future improvement

Can create this class for each dataset.
'''

class Dataset:

    def __init__(self, batch_size = 128, flatten = False):
        self._batch_size = batch_size
        if flatten:
            self._train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.flatten(x))])
            self._test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.flatten(x))])
        else:
            self._train_transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self._test_transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    def fetch():
        raise NotImplementedError

    @property
    def train_loader(self):
        if not hasattr(self, '_trainset'):
            raise NameError('You do not have trainset. Call fetch() first.')
        if not self._batch_size:
            self._batch_size = self._trainset.data.shape[0]
        return torch.utils.data.DataLoader(self._trainset, batch_size=self._batch_size, shuffle=True) 

    @property
    def test_loader(self):
        if not hasattr(self, '_testset'):
            raise NameError('You do not have testset. Call fetch() first.')
        if not self._batch_size:
            self._batch_size = self._testset.data.shape[0]
        return torch.utils.data.DataLoader(self._testset, batch_size=self._batch_size, shuffle=False)
        
    @property
    def num_classes(self):
        if not hasattr(self, '_trainset'):
            raise NameError('You do not have trainset/testset yet. Call fetch() first')
        return len(self._trainset.classes)

    @property
    def input_size(self):
        if not hasattr(self, '_trainset'):
            raise NameError('You do not have trainset/testset yet. Call fetch() first')
        shape = self._trainset.data[0].shape
        return functools.reduce(lambda a, b: a * b, shape)