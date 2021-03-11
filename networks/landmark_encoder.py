import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn
from nets import LinearBlock

class LandmarkEncoder(nn.Module):
    def __init__(self, input_dims = 6, start_hidden_dims = 256, n_layers = 2, device = 'cpu'):
        super(LandmarkEncoder, self).__init__()
        layers = [LinearBlock(input_dims, start_hidden_dims)]
        dims = start_hidden_dims
        for i in range(n_layers - 2):
            layers.append(LinearBlock(dims, dims*2))
            dims *= 2
        layers.extend([
            nn.Linear(dims, dims*2),
            nn.BatchNorm1d(dims*2),
            nn.Tanh(),
        ])
        self.__layers = nn.Sequential(*layers).to(device)
        self.__device = device

    def forward(self, x):
        '''
        x: (batchsize, features) -> default (batchsize, 6)
        output: (batchsize, features*)
        '''
        x = x.to(self.__device)
        return self.__layers(x)

if __name__ == "__main__":
    model = LandmarkEncoder()
    x = torch.ones(10, 6)
    y = model(x)
    print(y.shape)
