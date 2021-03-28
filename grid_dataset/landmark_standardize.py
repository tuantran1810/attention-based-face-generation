import cv2, sys, os, copy, math
import matplotlib.pyplot as plt
import numpy as np
from pickle import dump, load
from tqdm import tqdm

class LandmarkStandardize(object):
    def __init__(
        self,
        inputlandmark = "./preprocessed/raw_landmark.pkl",
        outputpath = "./preprocessed/standard_landmark.pkl",
    ):
        '''
        self.__paths: map[identity]pathsList
        '''
        self.__outputpath = outputpath
        self.__input_map = {}
        with open(inputlandmark, 'rb') as fd:
            self.__input_map = load(fd)
        self.__mean_shape = self.__input_map['mean']
        del self.__input_map['mean']

    def __transfer_expression(self, landmark_sequence):
        first_landmark = landmark_sequence[0,:,:]
        transform, _ = cv2.estimateAffine2D(first_landmark, self.__mean_shape, True)

        sx = np.sign(transform[0,0])*np.sqrt(transform[0,0]**2 + transform[0,1]**2)
        sy = np.sign(transform[1,0])*np.sqrt(transform[1,0]**2 + transform[1,1]**2)

        zero_vector = np.zeros((1, 68, 2))
        diff = np.cumsum(np.insert(np.diff(landmark_sequence, n=1, axis=0), 0, zero_vector, axis=0), axis=0)
        mean_shape_seq = np.tile(np.reshape(self.__mean_shape, (1, 68, 2)), [landmark_sequence.shape[0], 1, 1])

        diff[:, :, 0] = abs(sx)*diff[:, :, 0]
        diff[:, :, 1] = abs(sy)*diff[:, :, 1]

        transfer_expression_seq = diff + mean_shape_seq
        return np.float32(transfer_expression_seq)

    def run(self):
        for identity, identity_map in tqdm(self.__input_map.items()):
            for code, video_path in tqdm(identity_map.items()):
                landmarks = self.__input_map[identity][code]
                output_landmarks = self.__transfer_expression(landmarks)
                self.__input_map[identity][code] = output_landmarks
        with open(self.__outputpath, 'wb') as fd:
            dump(self.__input_map, fd)

def main():
    d = LandmarkStandardize(
        inputlandmark = "/media/tuantran/raid-data/dataset/GRID/raw_landmark.pkl",
        outputpath = "/media/tuantran/raid-data/dataset/GRID/standard_landmark.pkl",
    )
    d.run()

if __name__ == "__main__":
    main()
