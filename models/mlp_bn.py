import torch.nn as nn
from .mlp import MLP

class MLPBN(MLP):
    """ Multilayer Perceptron with batch normalization """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bn  = nn.BatchNorm1d(kwargs['num_units'])

    def forward(self, x):
        # first layer
        x = self.relu(self.bn(self.fc_input(x)))
        # middle layer(s)
        for i in range(self._num_hidden_layers):
            x = self.relu(self.bn(self.fc_middle(x)))
        # output layer
        x = self.fc_out(x)
        return x