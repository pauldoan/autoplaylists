# importing DL libraries
import torch.nn as nn
import torch.nn.functional as F


# NN model class
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation=F.relu, dropout=0.1):
        super(NeuralNet, self).__init__()
        
        # I want my model to be flexible, so I will allow for multiple hidden layers
        layers = []
        
        # Input layer to first hidden layer, hidden_sizes len should be at least 1
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.Dropout(dropout)) 
        
        # Adding hidden layers
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Store the layers in a ModuleList
        self.layers = nn.ModuleList(layers)

        # store the activation function
        self.activation = activation

    # forward pass
    def forward(self, x):
        # Pass input through each layer in the network
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)

            # we add the activation if layer is not dropout
            if isinstance(self.layers[i], nn.Linear):
                x = self.activation(x)
        
        # no activation for the last layer
        x = self.layers[-1](x)
        
        return x