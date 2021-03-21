import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn
from nets import Conv2dBlock, LinearBlock

class Discriminator(nn.Module):
    def __init__(self, 
        image_size = (128, 128), 
        image_channels = 3,
        image_encoder_conv_layers = 4,
        landmark_shape = (68, 2),
        device = 'cpu',
    ):
        super(Discriminator ,self).__init__()
        channels = 64
        image_w, image_h = image_size
        image_w //= 2**image_encoder_conv_layers
        image_h //= 2**image_encoder_conv_layers

        layers = [Conv2dBlock(in_channels = 3, out_channels = 64, kernel = 3, stride = 2, padiing = 1)]
        for _ in range(image_encoder_conv_layers):
            layers.append(Conv2dBlock(in_channels = channels, out_channels = channels*2, kernel = 3, stride = 2, padiing = 1))
            channels *= 2
        self.__image_encoder = nn.Sequential(*layers).to(device)
        self.__image_fc = nn.LinearBlock(image_w*image_h*channels, channels).to(device)

        landmark_points, landmark_dims = landmark_shape
        features = landmark_points * landmark_dims
        self.__landmark_encoder = nn.Sequential(
            LinearBlock(features, 256),
            LinearBlock(256, 512),
        )

        self.__lstm = nn.LSTM(input_size = 1024, hidden_size = 256, num_layers = 3, batch_first = True)
        
        self.__device = device

    def forward(self, generated_images, original_landmark):
        '''
        generated_images: (batchsize, channels, frames, w, h) -> default: (batchsize, 3, t, 128, 128)
        original_landmark: (batchsize, points, dims) -> default: (batchsize, 68, 2)
        output: (batchsize) -> default: (batchsize)
        '''
        generated_images = generated_images.to(self.__device)
        original_landmark = original_landmark.to(self.__device)
        batchsize = generated_images.shape[0]
        frames = generated_images.shape[2]

        landmark_features = self.__landmark_encoder(original_landmark)
        lstm_input = []
        for i in range(frames):
            img = generated_images[:,:,i,:,:]
            img = self.__image_encoder(img)
            img = img.view(batchsize, -1)
            img = self.__image_fc(img)
            feature = torch.cat([landmark_features, img], dim = 1)
            lstm_input.append(feature)
        lstm_input = torch.stack(lstm_input, dim = 1)
        lstm_output, _ = self.__lstm(lstm_input)
