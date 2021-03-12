import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn
from mfcc_encoder import MFCCEncoder
from landmark_encoder import LandmarkEncoder
import utils

class LandmarkDecoder(nn.Module):
    def __init__(self, mfcc_f = 12, mfcc_stride = 4, mfcc_window = 28, landmark_dims = 6, device = 'cpu'):
        super(LandmarkDecoder, self).__init__()
        self.__device = device
        self.__mfcc_encoder = MFCCEncoder(input_shape = (mfcc_f, mfcc_window)).to(device)
        self.__landmark_encoder = LandmarkEncoder(input_dims = landmark_dims).to(device)
        self.__lstm = nn.LSTM(input_size = 256+512, hidden_size = 256, num_layers = 3, batch_first = True).to(device)
        self.__fc = nn.Sequential(
            nn.Linear(256, landmark_dims),
            nn.BatchNorm1d(landmark_dims),
            nn.Tanh(),
        ).to(device)
        self.__mfcc_stride = mfcc_stride
        self.__mfcc_window = mfcc_window
        if mfcc_window % mfcc_stride != 0:
            raise Exception("invalid mfcc config")

    def forward(self, landmark, mfcc):
        '''
        landmark: (batchsize, points) -> default: (batchsize, 6)
        mfcc: (batchsize, channels, f, t) -> default: (batchsize, 1, 12, t)
        output: (batchsize, points*) -> default: (batchsize, t, 6)
        '''
        landmark = landmark.to(self.__device)
        mfcc = mfcc.to(self.__device)
        landmark_feature = self.__landmark_encoder(landmark)

        batchsize, _, mfcc_f, mfcc_t = mfcc.shape
        if mfcc_t % self.__mfcc_stride != 0:
            raise Exception("invalid mfcc_t")
        t_frames = mfcc_t//self.__mfcc_stride
        w_frames = self.__mfcc_window//self.__mfcc_stride
        t_padding = w_frames//2 * self.__mfcc_stride
        parts = (
            torch.ones(batchsize, 1, mfcc_f, t_padding),
            mfcc,
            torch.ones(batchsize, 1, mfcc_f, t_padding),
        )
        mfcc = torch.cat(parts, dim = 3)
        mfcc_array = []
        for i in range(t_frames):
            offset = i*self.__mfcc_stride
            chunk = mfcc[:,:,:,offset:(offset + self.__mfcc_window)]
            chunk = self.__mfcc_encoder(chunk)
            chunk = torch.cat([chunk, landmark_feature], dim = 1)
            chunk = chunk.unsqueeze(1)
            mfcc_array.append(chunk)
        mfcc_array = torch.cat(mfcc_array, dim = 1)
        out, _ = self.__lstm(mfcc_array)
        out_array = []
        for i in range(t_frames):
            t_feature = out[:,i,:]
            t_feature = self.__fc(t_feature)
            out_array.append(t_feature.unsqueeze(1))
        out_array = torch.cat(out_array, dim = 1)
        return out_array

if __name__ == "__main__":
    ldec = LandmarkDecoder()
    landmark = torch.ones(7, 6)
    mfcc = torch.ones(7, 1, 12, 75*4)
    out = ldec(landmark, mfcc)
    print(out.shape)
