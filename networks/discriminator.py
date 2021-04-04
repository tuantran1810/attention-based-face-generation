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

        layers = [Conv2dBlock(in_channels = 3, out_channels = 64, kernel = 3, stride = 2, padding = 1)]
        for _ in range(image_encoder_conv_layers - 1):
            layers.append(Conv2dBlock(in_channels = channels, out_channels = channels*2, kernel = 3, stride = 2, padding = 1))
            channels *= 2
        self.__image_encoder = nn.Sequential(*layers).to(device)
        self.__image_fc = LinearBlock(image_w*image_h*channels, channels).to(device)

        landmark_points, landmark_dims = landmark_shape
        features = landmark_points * landmark_dims
        self.__landmark_encoder = nn.Sequential(
            LinearBlock(features, 256),
            LinearBlock(256, 512),
        )

        self.__lstm = nn.LSTM(input_size = 1024, hidden_size = 256, num_layers = 3, batch_first = True)
        self.__lstm_fc = LinearBlock(256, landmark_points*landmark_dims, norm = None, activation = nn.Tanh)
        self.__decision_fc = nn.Linear(256, 1)
        self.__activation = nn.Sigmoid()
        
        self.__device = device

    def forward(self, generated_images, original_landmark):
        '''
        generated_images: (batchsize, channels, frames, w, h) -> default: (batchsize, 3, t, 128, 128)
        original_landmark: (batchsize, points, dims) -> default: (batchsize, 68, 2)
        output: tuple((batchsize), (batchsize, t, points, dims)) -> default: ((batchsize), (batchsize, t, 68, 2))
        '''
        generated_images = generated_images.to(self.__device)
        original_landmark = original_landmark.to(self.__device)
        batchsize, landmark_points, landmark_dims = original_landmark.shape
        frames = generated_images.shape[2]

        original_landmark = original_landmark.view(batchsize, -1)
        landmark_features = self.__landmark_encoder(original_landmark)

        lstm_input = list()
        for i in range(frames):
            img = generated_images[:,:,i,:,:]
            img = self.__image_encoder(img)
            img = img.view(batchsize, -1)
            img = self.__image_fc(img)
            feature = torch.cat([landmark_features, img], dim = 1)
            lstm_input.append(feature)
        lstm_input = torch.stack(lstm_input, dim = 1)
        lstm_output, _ = self.__lstm(lstm_input)

        landmark_output = list()
        decision_output = list()
        for i in range(frames):
            lstm_out = lstm_output[:,i,:]
            landmark_features = self.__lstm_fc(lstm_out)
            landmark = (landmark_features + original_landmark).view(batchsize, landmark_points, landmark_dims)
            landmark_output.append(landmark)
            decision = self.__decision_fc(lstm_out)
            decision_output.append(decision)
        landmark_output = torch.stack(landmark_output, dim = 1)
        decision_output = torch.stack(decision_output, dim = 2)
        decision_output = nn.functional.avg_pool1d(decision_output, kernel_size = frames)
        decision_output = self.__activation(decision_output)
        decision_output = decision_output.view(batchsize)
        return decision_output, landmark_output

if __name__ == "__main__":
    dis = Discriminator()
    gen_images = torch.ones(2, 3, 75, 128, 128)
    orig_landmark = torch.ones(2, 68, 2)
    decision_output, landmark_output = dis(gen_images, orig_landmark)
    print(decision_output.shape)
    print(landmark_output.shape)
