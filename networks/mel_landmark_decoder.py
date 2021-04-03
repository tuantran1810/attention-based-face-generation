import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn
from mel_encoder import MelEncoder
from landmark_encoder import PCALandmarkEncoder
from noise_lstm import NoiseLSTM

class MelLandmarkDecoder(nn.Module):
    def __init__(self, mel_f = 128, mel_t = 5, device = 'cpu'):
        super(MelLandmarkDecoder, self).__init__()
        self.__device = device
        self.__mel_encoder = MelEncoder(input_shape = (mel_t, mel_f), device = device)
        self.__mel_lstm_1 = nn.LSTM(input_size = 256, hidden_size = 256, num_layers = 3, batch_first = True, dropout = 0.2).to(device)
        self.__mel_lstm_2 = nn.LSTM(input_size = 256, hidden_size = 136, num_layers = 1, batch_first = True, dropout = 0.2).to(device)

        self.__mel_f = mel_f
        self.__mel_t = mel_t

    def forward(self, mel):
        '''
        mel: (batchsize, channels, f, t) -> default: (batchsize, 1, t, 128)
        output: (batchsize, points*) -> default: (batchsize, t, 136)
        '''
        mel = mel.to(self.__device)
        total_t = mel.shape[2]
        mel_array = []
        for i in range(0, total_t - self.__mel_t + 1):
            chunk = mel[:,:,i:i+self.__mel_t]
            chunk = self.__mel_encoder(chunk)
            chunk = chunk.unsqueeze(1)
            mel_array.append(chunk)
        mel_array = torch.cat(mel_array, dim = 1)

        hidden = ( torch.autograd.Variable(torch.zeros(3, mel.size(0), 256).cuda()),
                      torch.autograd.Variable(torch.zeros(3, mel.size(0), 256).cuda()))
        out, _ = self.__mel_lstm_1(mel_array, hidden)

        hidden = ( torch.autograd.Variable(torch.zeros(1, mel.size(0), 136).cuda()),
                      torch.autograd.Variable(torch.zeros(1, mel.size(0), 136).cuda()))
        out, _ = self.__mel_lstm_2(out, hidden)
        
        return out

if __name__ == "__main__":
    ldec = MelLandmarkDecoder()
    mel = torch.ones(7, 1, 75, 128)
    out = ldec(mel)
    print(out.shape)
