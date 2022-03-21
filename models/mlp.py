import torch.nn as nn

class MLP(nn.Module):
    """ Multilayer Perceptron """
    def __init__(self, input_size, num_hidden_layers, num_units, num_classes):
        super(MLP, self).__init__()
        self.fc_input  = nn.Linear(input_size, num_units)
        self.fc_middle = nn.Linear(num_units,  num_units)
        self.fc_out    = nn.Linear(num_units,  num_classes) 
        self.relu = nn.ReLU()
        self._num_hidden_layers = num_hidden_layers

    def forward(self, x):
        # first layer
        x = self.relu(self.fc_input(x))
        # middle layer(s)
        for i in range(self._num_hidden_layers):
            x = self.relu(self.fc_middle(x))
        # output layer
        x = self.fc_out(x)
        return x