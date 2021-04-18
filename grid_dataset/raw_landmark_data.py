import cv2, sys, os, copy, math, librosa, scipy.io.wavfile
sys.path.append(os.path.dirname(__file__))
from os import path
from pathlib import Path
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from mobilenet.mobile_net import MobileNetInference224
import numpy as np
from pickle import dump
from compress_pickle import load
from tqdm import tqdm

class RawLandmarkProcessor(object):
    def __init__(self, model_path = './mobilenet/mobilenet_224_model_best_gdconv_external.pth.tar', device = "cpu"):
        self.__landmark_detector = MobileNetInference224(
            model_path = model_path,
            device = device,
        )
        self.__landmark_size = 224

    def __resize_batch(self, frames, size):
        tmp_frames = []
        for frame in frames:
            frame = cv2.resize(frame, size)
            frame = np.expand_dims(frame, 0)
            tmp_frames.append(frame)
        frames = np.concatenate(tmp_frames, axis = 0)
        return frames

    def get_landmark(self, frames):
        frames = self.__resize_batch(frames.transpose(0,2,3,1), (self.__landmark_size, self.__landmark_size)).transpose(0,3,1,2)
        return self.__landmark_detector.infer_numpy(frames)

class RawLandmarkData(object):
    def __init__(
        self,
        imagerootfolder = "./preprocessed/images",
        landmark_outputpath = "./preprocessed/raw_landmark.pkl",
        image_ext = "gzip",
        device = 'cpu',
    ):
        '''
        self.__paths: map[identity]pathsList
        '''

        self.__device = device
        self.__paths = dict()

        for path, _ , files in os.walk(imagerootfolder):
            identity = path.split('/')[-1]
            imagemap = {}
            for name in files:
                code, file_ext = name.split('.')
                if file_ext == image_ext:
                    imagemap[code] = os.path.join(path, name)
            if len(imagemap) > 0:
                self.__paths[identity] = imagemap

        self.__landmark_outputpath = landmark_outputpath
        self.__landmark_processor = RawLandmarkProcessor(device = device)

    def run(self):
        all_landmarks = {}
        mean_landmark_array = []
        for identity, identity_map in tqdm(self.__paths.items()):
            identity_path = os.path.join(self.__landmark_outputpath, identity)
            all_landmarks[identity] = {}
            for code, image_path in tqdm(identity_map.items()):
                frames = None
                with open(image_path, 'rb') as fd:
                    data = load(fd, compression = 'gzip')
                    frames = data['faces']
                if frames is None:
                    print("invalid video: {}".format(image_path))
                    continue
                orig_landmarks = self.__landmark_processor.get_landmark(frames)
                all_landmarks[identity][code] = orig_landmarks
                mean_lm = np.mean(orig_landmarks, axis = 0)
                mean_landmark_array.append(mean_lm)
        mean_landmark_array = np.stack(mean_landmark_array)
        avg_landmark = np.mean(mean_landmark_array, axis = 0)
        all_landmarks['mean'] = avg_landmark
        with open(self.__landmark_outputpath, 'wb') as fd:
            dump(all_landmarks, fd)

def main():
    d = RawLandmarkData(
        imagerootfolder = "/media/tuantran/rapid-data/dataset/GRID/face_images_128",
        landmark_outputpath = "/media/tuantran/raid-data/dataset/GRID/raw_landmark_2.pkl",
        device = 'cuda:0',
    )
    d.run()

if __name__ == "__main__":
    main()
