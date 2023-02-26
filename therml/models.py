# Model
import numpy as np
import torch
import torch.nn as nn

# Single hidden layer NN
class Modelv1(nn.Module):

    def __init__(self, input_dim, units=1000, layers=1):
        super(Modelv1, self).__init__()
        self.input_dim = input_dim
        self.units = units
        self.layers = layers
        self.hidden_1 = nn.Linear(self.input_dim, self.units)
        if self.layers > 1:
            self.linears = nn.ModuleList([nn.Linear(self.units, self.units) for _ in range(self.layers - 1)])
        self.out = nn.Linear(self.units,1)
        self.relu = nn.ReLU()

    def forward(self, input1, input2, extrafeat):
        x_combined = torch.cat([input1, input2, extrafeat], axis=1)
        x_combined = self.relu(self.hidden_1(x_combined))
        if self.layers > 1:
            for layer in self.linears:
                x_combined = self.relu(layer(x_combined))
        return self.out(x_combined)

# Siamese NN
class ModelSiamese(nn.Module):

    def __init__(self, input_dim, extra_dims, units=1000, layers=1):
        super(ModelSiamese, self).__init__()
        self.input_dim = input_dim
        self.extra_dims = extra_dims
        self.units = units
        self.layers = layers
        self.hidden_1 = nn.Linear(self.input_dim, self.units)
        if self.layers > 1:
            self.net = nn.ModuleList([nn.Linear(self.units, self.units) for _ in range(self.layers - 1)])
        self.fc = nn.Linear(2*self.units + self.extra_dims, 1)
        self.relu = nn.ReLU()

    def forward(self, input1, input2, extrafeat):
        x1 = self.relu(self.hidden_1(input1))
        x2 = self.relu(self.hidden_1(input2))
        if self.layers > 1:
            for layer in self.net:
                x1 = self.relu(layer(x1))
                x2 = self.relu(layer(x2))
        x = torch.cat([x1,x2,extrafeat], axis=1)
        return self.fc(x)

def reset_weight(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
