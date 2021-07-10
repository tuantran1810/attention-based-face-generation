import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn
from nets import LinearBlock, Conv2dBlock, Deconv2dBlock, Resnet2dBlock
from image_encoder import ImageEncoder
from pytorch_convolutional_rnn.convolutional_rnn import Conv2dGRU

class Generator(nn.Module):
    def __init__(self, landmark_input_dims = (68, 2), landmark_hidden_shape = (8, 8), resnet_layers = 9, device = 'cpu'):
        super(Generator, self).__init__()
        points, pdims = landmark_input_dims
        landmark_hidden_dims = landmark_hidden_shape[0] * landmark_hidden_shape[1]
        self.__landmark_hidden_shape = landmark_hidden_shape
        self.__image_encoder = ImageEncoder(device = device)
        self.__landmark_first_stage_enc = LinearBlock(points*pdims, landmark_hidden_dims).to(device)
        self.__landmark_second_stage_enc = Conv2dBlock(1, 256, kernel = 3, stride = 1, padding = 1).to(device)
        self.__landmark_final_stage_enc = Conv2dBlock(256, 512, kernel = 3, stride = 1, padding = 1, activation = nn.Tanh).to(device)
        self.__landmark_attention = nn.Sequential(
            Deconv2dBlock(in_channels = 512, out_channels = 256, kernel = 3, stride = 2, padding = 1, output_padding = 1),
            Deconv2dBlock(in_channels = 256, out_channels = 128, kernel = 3, stride = 2, padding = 1, output_padding = 1),
            Conv2dBlock(in_channels = 128, out_channels = 1, kernel = 3, stride = 1, padding = 1, norm = None, activation = nn.Sigmoid),
        ).to(device)
        self.__bottle_neck = Conv2dBlock(in_channels = 1024, out_channels = 128, kernel = 3, stride = 1, padding = 1).to(device)
        self.__gru = Conv2dGRU(
            in_channels = 128,
            out_channels = 512,
            kernel_size = 3,
            num_layers = 1,
            bidirectional = False,
            dilation = 2,
            stride = 1,
            dropout = 0.5,
            batch_first = True,
        ).to(device)

        resnet_chain = [Resnet2dBlock(channels = 512, use_bias = False)] * resnet_layers
        resnet_chain.extend([
            Deconv2dBlock(in_channels = 512, out_channels = 256, kernel = 3, stride = 2, padding = 1, output_padding = 1),
            Deconv2dBlock(in_channels = 256, out_channels = 128, kernel = 3, stride = 2, padding = 1, output_padding = 1),
        ])
        self.__resnet_deconv = nn.Sequential(*resnet_chain).to(device)

        self.__final_deconv = nn.Sequential(
            Deconv2dBlock(in_channels = 128, out_channels = 64, kernel = 3, stride = 2, padding = 1, output_padding = 1),
            Deconv2dBlock(in_channels = 64, out_channels = 32, kernel = 3, stride = 2, padding = 1, output_padding = 1),
        ).to(device)

        self.__color_generator = Conv2dBlock(32, 3, kernel = 7, padding = 3, activation = nn.Tanh).to(device)
        self.__attention_generator = Conv2dBlock(32, 1, kernel = 7, padding = 3, activation = nn.Sigmoid).to(device)
        self.__device = device

    def forward(self, original_image, original_landmark, generated_landmark):
        '''
        original_image: (batchsize, channels, w, h) -> default (batchsize, 3, 128, 128)
        original_landmark: (batchsize, points) -> default (batchsize, 136)
        generated_landmark: (batchsize, frames, points) -> default (batchsize, frames, 136)
        output: 
            attention_map: (batchsize, channels, frames, w, h) -> default (batchsize, 1, frames, 128, 128)
            color_images: (batchsize, channels, frames, w, h) -> default (batchsize, 3, frames, 128, 128)
            final_images: (batchsize, channels, frames, w, h) -> default (batchsize, 3, frames, 128, 128)
        '''
        original_image = original_image.to(self.__device)
        original_landmark = original_landmark.to(self.__device)
        generated_landmark = generated_landmark.to(self.__device)
        batchsize, frames, points = generated_landmark.shape

        early_image_features, image_features = self.__image_encoder(original_image)

        original_landmark_features = self.__landmark_first_stage_enc(original_landmark)
        original_landmark_features = original_landmark_features.view(batchsize, *self.__landmark_hidden_shape)
        original_landmark_features = original_landmark_features.unsqueeze(1)
        original_landmark_features_second_stage = self.__landmark_second_stage_enc(original_landmark_features)
        original_landmark_features_final_stage = self.__landmark_final_stage_enc(original_landmark_features_second_stage)

        generated_landmark = generated_landmark.view(-1, points)
        generated_landmark_features = self.__landmark_first_stage_enc(generated_landmark).view(batchsize, frames, *self.__landmark_hidden_shape)

        gru_input = []
        landmark_attention = []
        for i in range(frames):
            tmp = generated_landmark_features[:, i, :, :]
            tmp = tmp.unsqueeze(1)
            tmp = self.__landmark_second_stage_enc(tmp)
            att_feature = torch.cat((original_landmark_features_second_stage, tmp), dim = 1)
            att = self.__landmark_attention(att_feature)
            lmark = self.__landmark_final_stage_enc(tmp)

            img_feature = torch.cat((image_features, lmark - original_landmark_features_final_stage), dim = 1)
            img_feature = self.__bottle_neck(img_feature)
            gru_input.append(img_feature)
            landmark_attention.append(att)

        landmark_attention = torch.stack(landmark_attention, dim = 1)
        gru_input = torch.stack(gru_input, dim = 1)
        gru_output, _ = self.__gru(gru_input)

        att_map = []
        color = []
        output = []
        for i in range(frames):
            gru_t = gru_output[:,i,:,:,:]
            att_t = landmark_attention[:,i,:,:,:]
            gru_t = self.__resnet_deconv(gru_t)
            image_features_t = early_image_features * (1 - att_t) + gru_t * att_t
            image_features_t = self.__final_deconv(image_features_t)
            color_t = self.__color_generator(image_features_t)
            color.append(color_t)
            attention_map_t = self.__attention_generator(image_features_t)
            att_map.append(attention_map_t)
            output_t = attention_map_t*color_t + (1 - attention_map_t)*original_image
            output.append(output_t)

        return torch.stack(att_map, dim = 2), torch.stack(color, dim = 2), torch.stack(output, dim = 2)

if __name__ == "__main__":
    gen = Generator(device = 'cpu')
    batchsize = 2
    orig_image = torch.ones(batchsize, 3, 128, 128)
    orig_landmark = torch.ones(batchsize, 136)
    gen_landmark = torch.ones(batchsize, 75, 136)
    att, color, out = gen(orig_image.float(), orig_landmark.float(), gen_landmark.float())
    print(att.shape)
    print(color.shape)
    print(out.shape)
