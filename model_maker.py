#from torchvision.models import resnet18
import torch.nn as nn
from models.mlp import MLP
from models.mlp_bn import MLPBN
from models.resnet18 import resnet18
from models.resnet18_mixup import resnet18_mixup
from models.resnet18_dropblock import resnet18_dropblock
from models.vgg13 import vgg13

class ModelMaker:
    def __init__(self, architecture, dataset_type, option=None, *args, **kwargs):
        self._architecture = architecture
        self._dataset_type = dataset_type
        if self._is_mlp():
            self._set_params(**kwargs)
        self._make_model(option, *args, **kwargs)
    
    def _set_params(self, num_classes, *args, **kwargs):
        self._params = {
            'num_classes': num_classes,
            'input_size': input_size,
            'num_hidden_layers': 2, 
            'num_units': 1000,
        }

    def _is_mlp(self):
        return self._architecture.startswith('mlp')

    def _make_model(self, option=None, *args, **kwargs):
        if self._architecture == 'mlp':
            print('Use Multilayer Perceptron')
            self._model = MLP(**self._params)
        elif self._architecture == 'mlp-bn':
            print('Use MLP with batch normalization')
            self._model = MLPBN(**self._params)
        elif self._architecture == 'resnet18':
            print('Use resnet18')
            if self._dataset_type in ['cifar10', 'svhn']:
                if option == None:
                    self._model = resnet18() # This is not from torchvision
                elif option == 'mixup':
                    self._model = resnet18_mixup(**kwargs)
                elif option == 'dropblock':
                    self._model = resnet18_dropblock(**kwargs)
            else:
                raise ValueError('Resnet for the dataset is not supported yet')
        elif self._architecture == 'vgg13':
            print('Use VGG13')
            if self._dataset_type in ['cifar10', 'svhn']:
                if option == None:
                    print("YeeeeeeY")
                    self._model = vgg13() # This is not from torchvision
        else:
            raise ValueError('Invalid model')

    @property
    def model(self):
        if not self._model:
            raise NameError('No model exists. You have to make model first')
        return self._model
