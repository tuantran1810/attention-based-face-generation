import os
import numpy as np
from sklearn.decomposition import PCA
from pickle import dump, load
from tqdm import tqdm

class DataPCACalculator(object):
    def __init__(
        self,
        landmarks_data = "./preprocessed/raw_landmark.pkl",
        landmark_mean_path = "./preprocessed/landmark_mean.pkl",
        output_file = "./preprocessed/landmark_pca.pkl"
    ):
        '''
        self.__paths: map[identity]pathsList
        '''

        self.__landmark_map = None
        with open(landmarks_data, 'rb') as fd:
            self.__landmark_map = load(fd)

        landmark_mean = None
        with open(landmark_mean_path, 'rb') as fd:
            landmark_mean = load(fd)
        if landmark_mean is None:
            raise Exception("cannot load pca metadata")
        self.__landmark_mean = landmark_mean

        self.__output_file = output_file

    def __pca_map(self, data, components):
        pca = PCA(n_components = components, copy = True)
        final = pca.fit(data)
        mp = {}
        mp["components"] = final.components_
        mp["explained_variance"] = final.explained_variance_
        mp["explained_variance_ratio"] = final.explained_variance_ratio_
        mp["singular_values"] = final.singular_values_
        mp["mean"] = final.mean_
        mp["n_components"] = final.n_components_
        mp["n_features"] = final.n_features_
        mp["n_samples"] = final.n_samples_
        mp["noise_variance"] = final.noise_variance_
        return mp

    def run(self):
        lst = []
        for k, lmp in self.__landmark_map.items():
            for _, landmarks in lmp.items():
                landmarks = landmarks - self.__landmark_mean
                frames = landmarks.shape[0]
                landmarks = landmarks.reshape(frames, -1)
                lst.append(landmarks)

        final = np.concatenate(lst, axis = 0)
        mp = {}
        for i in tqdm(range(6, 11), desc = 'PCA'):
            mp[i] = self.__pca_map(final, i)

        with open(self.__output_file, "wb") as fd:
            dump(mp, fd)


def main():
    d = DataPCACalculator(
        landmarks_data = "/media/tuantran/raid-data/dataset/GRID/standard_landmark.pkl",
        landmark_mean_path = "./preprocessed/landmark_mean.pkl",
        output_file = "./preprocessed/landmark_pca.pkl"
    )
    d.run()

if __name__ == "__main__":
    main()
