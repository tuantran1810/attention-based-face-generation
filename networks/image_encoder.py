import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn
from nets import Conv2dBlock

class ImageEncoder(nn.Module):
    def __init__(self, in_channels = 3, start_hidden_channels = 32, n_layers = 5, device = 'cpu'):
        super(ImageEncoder, self).__init__()
        n_second_conv_layers = n_layers//2
        n_first_conv_layers = n_layers - n_second_conv_layers
        channels = start_hidden_channels
        first_conv_layers = [
            nn.ReflectionPad2d(3),
            Conv2dBlock(
                in_channels = in_channels, 
                out_channels = channels,
                kernel = 7,
                stride = 1,
                padding = 0,
            ),
        ]

        for _ in range(n_first_conv_layers - 1):
            first_conv_layers.append(
                Conv2dBlock(
                    in_channels = channels,
                    out_channels = channels * 2,
                    kernel = 3,
                    stride = 2,
                    padding = 1,
                )
            )
            channels *= 2

        second_conv_layers = []
        for _ in range(n_second_conv_layers - 1):
            second_conv_layers.append(
                Conv2dBlock(
                    in_channels = channels,
                    out_channels = channels * 2,
                    kernel = 3,
                    stride = 2,
                    padding = 1,
                )
            )
            channels *= 2

        second_conv_layers.extend([
            nn.Conv2d(
                in_channels = channels,
                out_channels = channels * 2,
                kernel_size = 3,
                stride = 2,
                padding = 1,
            ),
            nn.BatchNorm2d(channels * 2),
            nn.Tanh(),
        ])

        self.__first_conv = nn.Sequential(*first_conv_layers).to(device)
        self.__second_conv = nn.Sequential(*second_conv_layers).to(device)
        self.__device = device

    def forward(self, x):
        '''
        x: (batchsize, channels, w, h) -> default: (batchsize, 3, 128, 128)
        output: tuple[(batchsize, *channels, *w, *h), (batchsize, **channels, **w, **h)] -> default: [(batchsize, 128, 32, 32), (batchsize, 512, 8, 8)]
        '''
        x = x.to(self.__device)
        y1 = self.__first_conv(x)
        y2 = self.__second_conv(y1)
        return (y1, y2)

if __name__ == '__main__':
    ienc = ImageEncoder()
    x = torch.ones(7, 3, 128, 128)
    y1, y2 = ienc(x)
    print(y1.shape)
    print(y2.shape)
