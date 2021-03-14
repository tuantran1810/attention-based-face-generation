import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn
from nets import LinearBlock, Conv2dBlock
from image_encoder import ImageEncoder

class Generator(nn.Module):
    def __init__(self, landmark_input_dims = (68, 2), landmark_hidden_shape = (8, 8), device = 'cpu'):
        super(LandmarkEncoder, self).__init__()
        points, pdims = input_dims
        landmark_hidden_dims = landmark_hidden_shape[0] * landmark_hidden_shape[1]
        self.__landmark_hidden_shape = landmark_hidden_shape
        self.__image_encoder = ImageEncoder(device = device)
        self.__landmark_first_stage_enc = LinearBlock(points*pdims, landmark_hidden_dims).to(device)
        self.__landmark_second_stage_enc = Conv2dBlock(1, 256, kernel = 3, stride = 1, padding = 1)
        self.__landmark_final_stage_enc = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.Tanh(),
        )
        self.__device = device

    def forward(self, original_image, original_landmark, generated_landmark):
        '''
        original_image: (batchsize, channels, w, h) -> default (batchsize, 3, 128, 128)
        original_landmark: (batchsize, points, dims) -> default (batchsize, 68, 2)
        generated_landmark: (batchsize, frames, points, dims) -> default (batchsize, frames, 68, 2)
        output: 
            attention_map: (batchsize, channels, frames, w, h) -> default (batchsize, frames, ..., ...)
            generated_landmark_features: (batchsize, features) -> default (batchsize, 64)
        '''
        original_image = original_image.to(self.__device)
        original_landmark = original_landmark.to(self.__device)
        generated_landmark = generated_landmark.to(self.__device)
        batchsize, frames, points, dims = generated_landmark.shape

        early_image_features, image_features = self.__image_encoder(original_image)

        generated_landmark = generated_landmark.view(batchsize, frames, -1)
        generated_landmark = generated_landmark.view(-1, points*dims)
        original_landmark = original_landmark.view(batchsize, -1)

        original_landmark_features = self.__landmark_first_stage_enc(original_landmark).view(batchsize, *self.__landmark_hidden_shape)
        original_landmark_features = original_landmark_features.unsqueeze(1)
        original_landmark_features = self.__landmark_second_stage_enc(original_landmark_features)
        original_landmark_features_dup = original_landmark_features.unsqueeze(2).repeat(1, 1, frames, 1, 1)

        generated_landmark_features = self.__landmark_first_stage_enc(generated_landmark).view(batchsize, frames, *self.__landmark_hidden_shape)
        generated_landmark_features = generated_landmark_features.view(batchsize, frames, *self.__landmark_hidden_shape).unsqueeze(1)

        tmp = []
        for i in range(generated_landmark_features.shape[2]):
            fr = generated_landmark_features[:,:,i,:,:]
            fr = self.__landmark_second_stage_enc(fr).unsqueeze(2)
            tmp.append(fr)
        generated_landmark_features = torch.cat(tmp, axis = 2)

        print(original_landmark_features_dup.shape)        
        print(generated_landmark_features.shape)        

        return 