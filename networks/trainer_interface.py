import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn
from torch.nn import functional as F
from landmark_decoder import LandmarkDecoder
from mel_landmark_decoder import MelLandmarkDecoder

class LandmarkDecoderTrainerInterface(LandmarkDecoder):
    def __init__(self, pca_dims, pca_mean, pca_components, device = "cpu"):
        super().__init__(landmark_dims = pca_dims, device = device)
        self.__pca_mean = pca_mean.to(device).unsqueeze(0)
        self.__pca_components = pca_components.transpose(0, 1).to(device)
        self.__device = device
    
    def forward(self, input_data):
        '''
        input_data: [batch_mfcc, batch_landmarks]
        batch_mfcc: (batchsize, channels, f, t) -> default: (batchsize, 1, 12, t)
        batch_landmarks: (batchsize, dims) -> default: (batchsize, 136)
        output: (batchsize, t, 6)
        '''
        mfcc, landmarks = input_data
        mfcc = mfcc.to(self.__device)
        landmarks = landmarks.to(self.__device)
        
        pca_landmarks = torch.matmul(landmarks - self.__pca_mean, self.__pca_components)
        return super().forward(pca_landmarks, mfcc)

class LandmarkMSELoss(nn.Module):
    def __init__(self, pca_mean, pca_components, y_padding, device = "cpu"):
        super(LandmarkMSELoss, self).__init__()
        self.__pca_mean = pca_mean.to(device).unsqueeze(0)
        self.__pca_components = pca_components.to(device)
        self.__device = device
        self.__y_padding = y_padding

    def forward(self, yhat, y):
        '''
        yhat: (batchsize, t, pca_dims) -> default: (batchsize, t, 6)
        y: (batchsize, t, original_dims) -> default: (batchsize, t', 68)
        output: loss score
        '''
        y = torch.matmul(y - self.__pca_mean, self.__pca_components)
        return F.mse_loss(yhat, y)

class MelLandmarkDecoderTrainerInterface(MelLandmarkDecoder):
    def __init__(self, dims, device = "cpu"):
        super().__init__(device = device)
        self.__device = device
    
    def forward(self, input_data):
        '''
        input_data: [batch_mel, batch_landmarks]
        batch_mel: (batchsize, channels, t, f) -> default: (batchsize, 1, t, 128)
        batch_landmarks: (batchsize, dims) -> default: (batchsize, 136)
        output: (batchsize, t, 136)
        '''
        mel, landmarks = input_data
        mel = mel.to(self.__device)
        # landmarks = landmarks.to(self.__device)

        return super().forward(mel)
