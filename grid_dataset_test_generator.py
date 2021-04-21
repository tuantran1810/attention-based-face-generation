import sys, os
sys.path.append(os.path.dirname(__file__))

import cv2
import torch
import ffmpeg
import pickle
import librosa
import random
from copy import deepcopy
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from networks.generator import Generator
from networks.landmark_decoder import LandmarkDecoder
from utils.media import vidwrite
import scipy.io.wavfile as wav
from compress_pickle import load as cpload

from grid_dataset.raw_face_data import RawFaceDataProcessor
from grid_dataset.raw_landmark_data import RawLandmarkProcessor
from grid_dataset.landmark_standardize import LandmarkTransformation
from grid_dataset.mfcc_data import MFCCProcessor

class GridGeneratorTest():
    def __init__(
        self,
        generator_path,
        generated_landmark_path,
        inspired_landmark_path,
        inspired_image_path,
        image_file_ext = "gzip",
        landmark_pca_path = "./grid_dataset/preprocessed/landmark_pca.pkl",
        landmark_pca_features = 7,
        standard_landmark_mean_path = "./grid_dataset/preprocessed/standard_landmark_mean.pkl",
        output_path = "./grid_generator_test_output/",
        device = 'cpu',
    ):
        self.__generator = Generator(device = device)
        self.__generator.load_state_dict(torch.load(generator_path))

        self.__generated_landmark_map = None
        with open(generated_landmark_path, 'rb') as fd:
            self.__generated_landmark_map = pickle.load(fd)

        self.__inspired_landmark_map = None
        with open(inspired_landmark_path, 'rb') as fd:
            self.__inspired_landmark_map = pickle.load(fd)

        self.__standard_landmark_mean = None
        with open(standard_landmark_mean_path, 'rb') as fd:
            self.__standard_landmark_mean = pickle.load(fd)

        self.__face_video_paths = dict()
        for path, _ , files in os.walk(inspired_image_path):
            identity = path.split('/')[-1]
            videomap = {}
            for name in files:
                code, file_ext = name.split('.')
                if file_ext == image_file_ext:
                    videomap[code] = os.path.join(path, name)
            if len(videomap) > 0:
                self.__face_video_paths[identity] = videomap

        pca_landmark_metadata = None
        with open(landmark_pca_path, 'rb') as fd:
            pca_landmark_metadata = pickle.load(fd)

        pca_landmark_metadata = pca_landmark_metadata[landmark_pca_features]
        self.__landmark_pca_mean = pca_landmark_metadata["mean"]
        self.__landmark_pca_components = pca_landmark_metadata["components"]

        self.__image_transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        self.__device = device
        self.__output_path = output_path
        self.__data = {}

    def set_input(self, input_map):
        name_list = []
        generated_landmarks_list = []
        inspired_landmark_list = []
        inspired_face_image_list = []
        for identity, code_list in input_map.items():
            for code in code_list:
                name = "{}_{}".format(identity, code)
                name_list.append(name)
                r = random.randint(6, 49)

                generated_pca_landmark = self.__generated_landmark_map[identity][code][r-6]
                generated_landmarks = np.matmul(generated_pca_landmark, self.__landmark_pca_components) + self.__landmark_pca_mean
                generated_landmarks = generated_landmarks.reshape(16, 68, 2) + self.__standard_landmark_mean
                generated_landmarks = generated_landmarks - 0.5

                generated_landmarks_list.append(generated_landmarks)

                face_landmark = self.__inspired_landmark_map[identity][code][r] - 0.5
                inspired_landmark_list.append(face_landmark)

                face_image_sequence = None
                image_path = self.__face_video_paths[identity][code]
                with open(image_path, 'rb') as fd:
                    face_image_sequence = cpload(fd, compression = 'gzip')
                face_image = face_image_sequence['faces'][r]
                face_image = face_image.transpose(1, 2, 0)
                face_image = self.__image_transform_ops(face_image)
                inspired_face_image_list.append(face_image)
        self.__data['name_list'] = name_list
        self.__data['generated_landmarks'] = torch.tensor(np.stack(generated_landmarks_list)).float().to(self.__device)
        self.__data['inspired_landmarks'] = torch.tensor(np.stack(inspired_landmark_list)).float().to(self.__device)
        self.__data['inspired_face_image'] = torch.stack(inspired_face_image_list).float().to(self.__device)

    def infer(self):
        generated_landmarks = self.__data['generated_landmarks']
        inspired_landmarks = self.__data['inspired_landmarks']
        inspired_face_image = self.__data['inspired_face_image']

        batchsize, frames, _, _ = generated_landmarks.shape
        generated_landmarks = generated_landmarks.view(batchsize, frames, -1)
        inspired_landmarks = inspired_landmarks.view(batchsize, -1)

        # self.__generator.eval()
        with torch.no_grad():
            att, color, final = self.__generator(inspired_face_image, inspired_landmarks, generated_landmarks)
            self.__data['output_att'] = att.detach().cpu().numpy()
            self.__data['output_color'] = np.uint8((color.detach().cpu().numpy() + 1.0) * 127.5)
            self.__data['output_final'] = np.uint8((final.detach().cpu().numpy() + 1.0) * 127.5)

    def to_video(self):
        name_list = self.__data['name_list']
        batch_att = self.__data['output_att']
        batch_color = self.__data['output_color']
        batch_final = self.__data['output_final']
        plain_video_folder_path = os.path.join(self.__output_path, 'plain_video')
        Path(plain_video_folder_path).mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(name_list):
            att = (np.repeat(batch_att[i].transpose(1,2,3,0), 3, axis = 3) * 255.0).astype(np.uint8)
            color = batch_color[i].transpose(1,2,3,0)
            final = batch_final[i].transpose(1,2,3,0)

            combined = np.concatenate(
                [
                    final,
                    color,
                    att,
                ],
                axis = 2,
            )

            video_frame_array = []
            for frame in combined:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_frame_array.append(frame)
            combined = np.stack(video_frame_array)

            plain_video_path = os.path.join(plain_video_folder_path, name + '.mp4')
            vidwrite(plain_video_path, combined, vcodec='libx264', fps=25)

def main():
    generator_path = "./model/grid/generator.pt"
    generated_landmark_path = "/media/tuantran/raid-data/dataset/GRID/attention-based-face-generation/generated_pca_landmark_6_50.pkl"
    inspired_landmark_path = "/media/tuantran/raid-data/dataset/GRID/standard_landmark.pkl"
    inspired_image_path = "/media/tuantran/rapid-data/dataset/GRID/face_images_128"

    input_map = {
        "s1": ["bwwbzp", "lgif1s", "prbj6p"],
        "s27": ["lgaj2n", "pbwa5a", "sbby9s"],
    }

    test = GridGeneratorTest(
        generator_path,
        generated_landmark_path,
        inspired_landmark_path,
        inspired_image_path,
        # device = "cuda:0"
    )

    test.set_input(input_map)
    test.infer()
    test.to_video()

if __name__ == "__main__":
    main()

