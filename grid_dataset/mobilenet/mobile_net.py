from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.models as models
from collections import namedtuple
import math
import pdb

##################################  Original Arcface Model #############################################################

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

##################################  MobileFaceNet #############################################################
    
class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Depth_Wise(Module):
    def __init__(self, in_c, out_c, residual = False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)
    def forward(self, x):
        return self.model(x)

class GNAP(Module):
    def __init__(self, embedding_size):
        super(GNAP, self).__init__()
        assert embedding_size == 512
        self.bn1 = BatchNorm2d(512, affine=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.bn2 = BatchNorm1d(512, affine=False)

    def forward(self, x):
        x = self.bn1(x)
        x_norm = torch.norm(x, 2, 1, True)
        x_norm_mean = torch.mean(x_norm)
        weight = x_norm_mean / x_norm
        x = x * weight
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        feature = self.bn2(x)
        return feature

class GDC(Module):
    def __init__(self, embedding_size):
        super(GDC, self).__init__()
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7,7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        #self.bn = BatchNorm1d(embedding_size, affine=False)
        self.bn = BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.conv_6_dw(x)
        x = self.conv_6_flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        return x

class MobileFaceNet(Module):
    def __init__(self, input_size, embedding_size = 512, output_name = "GDC"):
        super(MobileFaceNet, self).__init__()
        assert output_name in ["GNAP", 'GDC']
        assert input_size[0] in [112]
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        if output_name == "GNAP":
            self.output_layer = GNAP(512)
        else:
            self.output_layer = GDC(embedding_size)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        out = self.conv1(x)

        out = self.conv2_dw(out)

        out = self.conv_23(out)

        out = self.conv_3(out)

        out = self.conv_34(out)

        out = self.conv_4(out)

        out = self.conv_45(out)

        out = self.conv_5(out)

        conv_features = self.conv_6_sep(out)
        out = self.output_layer(conv_features)
        return out, conv_features

        return x

class MobileNetInference112:
    def __init__(self, model_path = "./mobilefacenet_model_best.pth.tar", device = 'cpu'):
        self.__std = torch.tensor([0.229, 0.224, 0.225]*112*112).reshape(112,112,3).transpose(2,0)
        self.__std = torch.unsqueeze(self.__std, 0).to(device)
        self.__mean = torch.tensor([0.485, 0.456, 0.406]*112*112).reshape(112,112,3).transpose(2,0)
        self.__mean = torch.unsqueeze(self.__mean, 0).to(device)
        self.__model = MobileFaceNet([112, 112], 136).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        self.__model.load_state_dict(checkpoint['state_dict'])
        self.__model.eval()
        self.__device = device

    def infer_numpy(self, image_batch):
        '''
        image_batch: shape(batch, channels, w = 112, h = 112), range: 0-255
        '''
        image_batch = torch.from_numpy(image_batch)
        result = self.infer(image_batch)
        result = result.to('cpu')
        return result.detach().numpy()

    def infer(self, image_batch):
        '''
        image_batch: shape(batch, channels, w = 112, h = 112), range: 0-255
        '''
        image_batch = image_batch.float()
        image_batch = image_batch.to(self.__device)
        batchsize = image_batch.shape[0]
        std = self.__std.repeat(batchsize, 1, 1, 1)
        mean = self.__mean.repeat(batchsize, 1, 1, 1)
        image_batch = image_batch/255.0
        image_batch = (image_batch - mean)/std
        result = self.__model(image_batch)[0]
        return result.reshape(batchsize, -1, 2)

class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)
            

# SE module
# https://github.com/wujiyang/Face_Pytorch/blob/master/backbone/cbam.py
class SEModule(nn.Module):
    '''Squeeze and Excitation Module'''
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return input * x
        
# USE global depthwise convolution layer. Compatible with MobileNetV2 (224×224), MobileNetV2_ExternalData (224×224)
class MobileNet_GDConv(nn.Module):
    def __init__(self,num_classes):
        super(MobileNet_GDConv,self).__init__()
        self.pretrain_net = models.mobilenet_v2(pretrained=False)
        self.base_net = nn.Sequential(*list(self.pretrain_net.children())[:-1])
        self.linear7 = ConvBlock(1280, 1280, (7, 7), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(1280, num_classes, 1, 1, 0, linear=True)
    def forward(self,x):
        x = self.base_net(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        return x

# USE global depthwise convolution layer. Compatible with MobileNetV2 (56×56)
class MobileNet_GDConv_56(nn.Module):
    def __init__(self,num_classes):
        super(MobileNet_GDConv_56,self).__init__()
        self.pretrain_net = models.mobilenet_v2(pretrained=False)
        self.base_net = nn.Sequential(*list(self.pretrain_net.children())[:-1])
        self.linear7 = ConvBlock(1280, 1280, (2, 2), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(1280, num_classes, 1, 1, 0, linear=True)
    def forward(self,x):
        x = self.base_net(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        return x        

 # MobileNetV2 with SE; Compatible with MobileNetV2_SE (224×224) and MobileNetV2_SE_RE (224×224)     
class MobileNet_GDConv_SE(nn.Module):
    def __init__(self,num_classes):
        super(MobileNet_GDConv_SE,self).__init__()
        self.pretrain_net = models.mobilenet_v2(pretrained=True)
        self.base_net = nn.Sequential(*list(self.pretrain_net.children())[:-1])
        self.linear7 = ConvBlock(1280, 1280, (7, 7), 1, 0, dw=True, linear=True) 
        self.linear1 = ConvBlock(1280, num_classes, 1, 1, 0, linear=True)
        self.attention=SEModule(1280,8)
    def forward(self,x):
        x = self.base_net(x)
        x = self.attention(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        return x

class MobileNetInference224:
    def __init__(self, model_path = "./mobilenet_224_model_best_gdconv_external.pth.tar", device = 'cpu'):
        self.__std = torch.tensor([0.229, 0.224, 0.225]*224*224).reshape(224,224,3).transpose(2,0)
        self.__std = torch.unsqueeze(self.__std, 0).to(device)
        self.__mean = torch.tensor([0.485, 0.456, 0.406]*224*224).reshape(224,224,3).transpose(2,0)
        self.__mean = torch.unsqueeze(self.__mean, 0).to(device)
        self.__model = torch.nn.DataParallel(MobileNet_GDConv(136).to(device))
        checkpoint = torch.load(model_path, map_location=device)
        self.__model.load_state_dict(checkpoint['state_dict'])
        self.__model.eval()
        self.__device = device

    def infer_numpy(self, image_batch):
        '''
        image_batch: shape(batch, channels, w = 224, h = 224), range: 0-255
        '''
        image_batch = torch.from_numpy(image_batch)
        result = self.infer(image_batch)
        result = result.to('cpu')
        return result.detach().numpy()

    def infer(self, image_batch):
        '''
        image_batch: shape(batch, channels, w = 224, h = 224), range: 0-255
        '''
        image_batch = image_batch.float()
        image_batch = image_batch.to(self.__device)
        batchsize = image_batch.shape[0]
        std = self.__std.repeat(batchsize, 1, 1, 1)
        mean = self.__mean.repeat(batchsize, 1, 1, 1)
        image_batch = image_batch/255.0
        image_batch = (image_batch - mean)/std
        result = self.__model(image_batch)
        return result.reshape(batchsize, -1, 2)
