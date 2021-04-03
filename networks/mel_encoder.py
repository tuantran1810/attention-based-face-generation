import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn
from nets import Conv2dBlock, LinearBlock
import net_utils

class MelEncoder(nn.Module):
    def __init__(self, in_channels = 1, start_hidden_channels = 8, input_shape = (5, 128), output_features = 256, device = 'cpu'):
        super(MelEncoder, self).__init__()
        f, t = input_shape
        conv_layers = [
            Conv2dBlock(in_channels = 1, out_channels = start_hidden_channels, kernel = 1, stride = 1, padding = 1),
        ]
        f, t = f+2, t+2
        kernel = 3
        stride = (1, 2)
        padding = 1
        channels = start_hidden_channels
        while True:
            if t <= 4:
                break
            conv_layers.append(
                Conv2dBlock(in_channels = channels, out_channels = channels*2, kernel = kernel, stride = stride, padding = padding),
            )
            if t > 4:
                t = net_utils.conv_output(size = t, kernel = kernel, stride = stride[1], padding = padding)
            channels *= 2

        features = channels * f * t
        if features < 1024:
            raise Exception("do not have enough features for linear layers")

        linear_layers = [
            LinearBlock(features, 1024, norm = None),
            LinearBlock(1024, 512, norm = None),
        ]
        linear_layers.extend([
            nn.Linear(512, output_features),
            nn.Tanh(),
        ])
        self.__device = device
        self.__conv_layers = nn.Sequential(*conv_layers).to(device)
        self.__linear_layers = nn.Sequential(*linear_layers).to(device)

    def forward(self, x):
        '''
        input: (batchsize, channels, t, f) -> default: (batchsize, 1, t, 128)
        output: (batchsize, features)
        '''
        x = x.to(self.__device)
        batchsize = x.shape[0]
        x = self.__conv_layers(x)
        x = x.view(batchsize, -1)        
        return self.__linear_layers(x)

if __name__ == "__main__":
    inp = torch.ones(7, 1, 5, 128)
    model = MelEncoder()
    y = model(inp)
    print(y.shape)
