import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn

class NoiseLSTM(nn.Module):
    def __init__(self, input_size = 10, output_size = 10, device = "cpu"):
        super(NoiseLSTM, self).__init__()
        self.__lstm = nn.LSTM(input_size = input_size, hidden_size = output_size, batch_first = True).to(device)
        self.__device = device

    def forward(self, x):
        '''
        x: (batch_size, t, values) ->
        output: (batch_size, t, values)
        '''
        x = x.to(self.__device)
        out, _ = self.__lstm(x)
        return out


if __name__ == "__main__":
    x = torch.ones(30, 75, 10)
    noise = NoiseLSTM(device = "cpu")
    x = noise(x)
    print(x.shape)