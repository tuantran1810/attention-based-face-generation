import sys, os
sys.path.append(os.path.dirname(__file__))

import cv2
import torch
import ffmpeg
import pickle
import librosa
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from networks.generator import Generator
from utils.media import vidwrite
import scipy.io.wavfile as wav
from skimage import transform as tf
from grid_dataset.raw_face_data import RawFaceDataProcessor
from grid_dataset.raw_landmark_data import RawLandmarkProcessor
from grid_dataset.landmark_standardize import LandmarkTransformation

class GridDemo:
    def __init__(
        self,
        generator_path,
        images_path,
        audios_path,
        output_path,
        raw_landmark_mean_path = "./grid_dataset/preprocessed/raw_landmark_mean.pkl",
        image_ext = "png",
        audio_ext = "wav",
        device = "cpu"
    ):
        self.__generator = Generator(device = device)
        self.__generator.load_state_dict(torch.load(generator_path))
        self.__raw_face_processor = RawFaceDataProcessor(device = device)
        self.__raw_landmark_processor = RawLandmarkProcessor(
            model_path = './grid_dataset/mobilenet/mobilenet_224_model_best_gdconv_external.pth.tar',
            device = device,
        )

        raw_landmark_mean = None
        with open(raw_landmark_mean_path, 'rb') as fd:
            raw_landmark_mean = pickle.load(fd)

        self.__landmark_transformation = LandmarkTransformation(raw_landmark_mean)
        self.__data_paths = {}
        for path, _ , files in os.walk(images_path):
            for file in files:
                parts = file.split('.')
                if len(parts) != 2:
                    continue
                filename, ext = parts
                if ext != image_ext:
                    continue
                self.__data_paths[filename] = {
                    "image": os.path.join(path, file),
                }

        for path, _ , files in os.walk(audios_path):
            for file in files:
                parts = file.split('.')
                if len(parts) != 2:
                    continue
                filename, ext = parts
                if ext != audio_ext:
                    continue
                if filename not in self.__data_paths:
                    print(f"no image for: {filename}")
                    continue
                self.__data_paths[filename]["audio"] = os.path.join(path, file)

        self.__data = {}

    def preprocess(self):
        for key, value in self.__data_paths.items():
            if 'image' not in value or 'audio' not in value:
                print(f"{key} does not have enough image/audio")
                continue
            img = cv2.imread(value['image'])
            img = np.expand_dims(img, 0)
            img = self.__raw_face_processor.crop_face(img)
            img = self.__raw_face_processor.resize_batch(img, (128,128)).transpose(0,3,1,2)

            landmark = self.__raw_landmark_processor.get_landmark(img)
            landmark = self.__landmark_transformation.transfer_single_landmark(landmark[0])
            landmark = np.expand_dims(landmark, 0)
            landmark = self.__landmark_transformation.align_eye_points(landmark)

            tform = tf.SimilarityTransform()
            mean_shape = self.__landmark_transformation.get_mean_shape()
            mean_shape_trans = mean_shape[27:45, :]
            landmark_trans = landmark[0,27:45,:]
            # img = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)
            # print(img.shape)
            # print(mean_shape_trans.shape)
            # print(landmark_trans.shape)
            tform.estimate(mean_shape_trans, landmark_trans)
            print(img.shape)
            img = tf.warp(img, tform)
            # img = np.uint8(img * 255)
            # landmark = self.__landmark_transformation.transfer_expression(landmark)
            plt.figure()
            plt.imshow(img.transpose(1,2,0))
            # plt.scatter(landmark[0,:,0], landmark[0,:,1])
        plt.show()
        plt.close()

def main():
    generator_path = "./model/grid/generator.pt"
    images_path = "./demo/images/"
    audios_path = "./demo/audios/"
    output_path = "./demo/output/"

    demo = GridDemo(
        generator_path,
        images_path,
        audios_path,
        output_path,
    )

    demo.preprocess()

if __name__ == "__main__":
    main()
