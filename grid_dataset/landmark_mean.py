import os
import numpy as np
import matplotlib.pyplot as plt
from pickle import dump, load
from tqdm import tqdm

class LandmarkMeanCalculator(object):
    def __init__(
        self,
        landmarks_data = "./preprocessed/raw_landmark.pkl",
        output_file = "./preprocessed/landmark_mean.pkl"
    ):
        '''
        self.__paths: map[identity]pathsList
        '''

        self.__landmark_map = None
        with open(landmarks_data, 'rb') as fd:
            self.__landmark_map = load(fd)
        self.__output_file = output_file

    def run(self):
        lst = []
        for k, lmp in self.__landmark_map.items():
            for _, landmarks in lmp.items():
                mlandmark = np.mean(landmarks, axis = 0)
                lst.append(mlandmark)

        final = np.stack(lst, axis = 0)
        final = np.mean(final, axis = 0)

        plt.figure()
        plt.scatter(final[:,0], final[:,1])
        plt.show()

        with open(self.__output_file, "wb") as fd:
            dump(final, fd)


def main():
    d = LandmarkMeanCalculator(
        landmarks_data = "/media/tuantran/raid-data/dataset/GRID/standard_landmark.pkl",
    )
    d.run()

if __name__ == "__main__":
    main()
