import sys, os
sys.path.append(os.path.dirname(__file__))
import math
import torch
from torch import nn
from nets import Conv2dBlock, LinearBlock
import utils

class MFCCEncoder(nn.Module):
    def __init__(self, in_channels = 1, start_hidden_channels = 32, input_shape = (12, 28), output_features = 6, device = 'cpu'):
        super(MFCCEncoder, self).__init__()
        f, t = input_shape
        conv_layers = [
            Conv2dBlock(in_channels = 1, out_channels = start_hidden_channels, kernel = 1, stride = 1, padding = 1),
        ]
        f, t = f+2, t+2
        kernel = 3
        stride_f = 2
        stride_t = 2
        padding = 1
        channels = start_hidden_channels
        while True:
            if f <= 4:
                stride_f = 1
            if t <= 4:
                stride_t = 1
            stride = (stride_f, stride_t)
            if stride == (1, 1):
                break
            conv_layers.append(
                Conv2dBlock(in_channels = channels, out_channels = channels*2, kernel = kernel, stride = stride, padding = padding),
            )
            if f > 4:
                f = utils.conv_output(size = f, kernel = kernel, stride = stride_f, padding = padding)
            if t > 4:
                t = utils.conv_output(size = t, kernel = kernel, stride = stride_t, padding = padding)
            channels *= 2

        features = channels * f * t
        if features < 1024:
            raise Exception("do not have enough feature for linear layers")

        linear_layers = [
            LinearBlock(features, 1024),
            LinearBlock(1024, 512),
            LinearBlock(512, 256),
        ]
        linear_layers.extend([
            nn.Linear(256, output_features),
            nn.BatchNorm1d(output_features),
            nn.Tanh(),
        ])
        self.__device = device
        self.__conv_layers = nn.Sequential(*conv_layers).to(device)
        self.__linear_layers = nn.Sequential(*linear_layers).to(device)

    def forward(self, x):
        '''
        input: (batchsize, channels, f, t) -> default: (batchsize, 1, 12, 28)
        output: (batchsize, features)
        '''
        x = x.to(self.__device)
        batchsize = x.shape[0]
        x = self.__conv_layers(x)
        x = x.view(batchsize, -1)        
        return self.__linear_layers(x)

if __name__ == "__main__":
    inp = torch.ones(7, 1, 12, 28)
    model = MFCCEncoder()
    y = model(inp)
    print(y.shape)
