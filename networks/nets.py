import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn

class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride = 1, padding = 0, norm = nn.BatchNorm1d, activation = nn.ReLU):
        super(Conv1dBlock, self).__init__()
        layers = [nn.Conv1d(in_channels, out_channels, kernel, stride, padding)]
        if norm is not None:
            layers.append(norm(out_channels))
        if activation is not None:
            layers.append(activation())
        self.__layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.__layers(x)

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride = 1, padding = 0, norm = nn.BatchNorm2d, activation = nn.ReLU):
        super(Conv2dBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel, stride, padding)]
        if norm is not None:
            layers.append(norm(out_channels))
        if activation is not None:
            layers.append(activation())
        self.__layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.__layers(x)

class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride = 1, padding = 0, norm = nn.BatchNorm3d, activation = nn.ReLU):
        super(Conv3dBlock, self).__init__()
        layers = [nn.Conv3d(in_channels, out_channels, kernel, stride, padding)]
        if norm is not None:
            layers.append(norm(out_channels))
        if activation is not None:
            layers.append(activation())
        self.__layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.__layers(x)

class Resnet2dBlock(nn.Module):
    def __init__(self, channels, use_dropout = True, use_bias = True):
        super(Resnet2dBlock, self).__init__()
        layers = [
            nn.ReplicationPad2d(1),
            nn.Conv2d(channels, channels, kernel_size = 3, padding = 0, bias = use_bias),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace = True),
        ]
        if use_bias:
            layers.append(nn.Dropout(0.5))
        layers.extend([
            nn.ReplicationPad2d(1),
            nn.Conv2d(channels, channels, kernel_size = 3, padding = 0, bias = use_bias),
            nn.BatchNorm2d(channels),
        ])

        self.blocks = nn.Sequential(*layers)
    
    def forward(self, x):
        return x + self.blocks(x)

class Resnet3dBlock(nn.Module):
    def __init__(self, channels, use_dropout = True, use_bias = True):
        super(Resnet3dBlock, self).__init__()
        layers = [
            nn.ReplicationPad3d(1),
            nn.Conv3d(channels, channels, kernel_size = 3, padding = 0, bias = use_bias),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace = True),
        ]
        if use_bias:
            layers.append(nn.Dropout(0.5))
        layers.extend([
            nn.ReplicationPad3d(1),
            nn.Conv3d(channels, channels, kernel_size = 3, padding = 0, bias = use_bias),
            nn.BatchNorm3d(channels),
        ])

        self.blocks = nn.Sequential(*layers)
    
    def forward(self, x):
        return x + self.blocks(x)

class Deconv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride = 1, padding = 0, dilation = 1, output_padding = 0, norm = nn.BatchNorm2d, activation = nn.ReLU):
        super(Deconv2dBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding, dilation = dilation, output_padding = output_padding)]
        if norm is not None:
            layers.append(norm(out_channels))
        if activation is not None:
            layers.append(activation())
        self.__layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.__layers(x)

class Deconv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride = 1, padding = 0, dilation = 1, output_padding = 0, norm = nn.BatchNorm3d, activation = nn.ReLU):
        super(Deconv3dBlock, self).__init__()
        layers = [nn.ConvTranspose3d(in_channels, out_channels, kernel, stride, padding, dilation = dilation, output_padding = output_padding)]
        if norm is not None:
            layers.append(norm(out_channels))
        if activation is not None:
            layers.append(activation())
        self.__layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.__layers(x)

class Unet2dBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, kernel, skip_kernel = 3, stride = 1, padding = 0, dilation = 1, output_padding = 0):
        super(Unet2dBlock, self).__init__()
        self.__layers = nn.Sequential(
            Deconv2dBlock(
                in_channels = in_channels + skip_channels, 
                out_channels = in_channels, 
                kernel = skip_kernel,
                stride = 1,
                padding = padding,
                output_padding = output_padding,
                dilation = dilation,
            ),
            Deconv2dBlock(
                in_channels = in_channels, 
                out_channels = out_channels, 
                kernel = kernel,
                stride = stride,
                padding = padding,
                output_padding = output_padding,
                dilation = dilation,
            ),
        )

    def forward(self, x, s):
        x = torch.cat([x, s], 1)
        x = self.__layers(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, input_dims, output_dims, norm = nn.BatchNorm1d, activation = nn.ReLU):
        super(LinearBlock, self).__init__()
        layers = [nn.Linear(input_dims, output_dims)]
        if norm is not None:
            layers.append(norm(output_dims))
        if activation is not None:
            layers.append(activation())
        self.__layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.__layers(x)

if __name__ == "__main__":
    unet = Unet2dBlock(
        in_channels = 64,
        skip_channels = 16,
        out_channels = 128,
        kernel = 4,
        stride = 2,
        padding = 1,
    )
    x = torch.ones(7, 64, 48, 64)
    s = torch.ones(7, 16, 48, 64)
    y = unet(x, s)
    print(y.shape)
