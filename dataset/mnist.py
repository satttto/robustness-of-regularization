import torchvision
import torchvision.transforms as transforms
from .base import Dataset

class MNIST(Dataset):
    
    def fetch(self):
        # fetch
        self._trainset = torchvision.datasets.MNIST(root='./data', train=True,  transform=self._train_transform, download=True)
        self._testset  = torchvision.datasets.MNIST(root='./data', train=False, transform=self._test_transform, download=True)    
        
    
