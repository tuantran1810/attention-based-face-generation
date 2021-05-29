import pickle
import numpy as np
from sklearn.decomposition import PCA
from pickle import dump, load
from tqdm import tqdm
import copy

class DataPCACalculator(object):
    def __init__(
        self,
        landmark_path = '/media/tuantran/raid-data/dataset/LRW/attention-based-face-generation/standard_landmark_mobinet.pkl',
        landmark_mean_path = "./preprocessed/standard_landmark_mean.pkl",
    ):
        landmark_mean = None
        with open(landmark_mean_path, 'rb') as fd:
            landmark_mean = load(fd)
        if landmark_mean is None:
            raise Exception("cannot load pca metadata")
        landmark_mean = landmark_mean.astype(np.float32)

        lst = []
        all_landmarks = None
        with open(landmark_path, 'rb') as fd:
            all_landmarks = pickle.load(fd)

        for utterance, utterance_map in all_landmarks.items():
            if 'train' not in utterance_map:
                print(f"there is no train in {utterance}")
                continue
            code_map = utterance_map['train']
            for _, landmarks in code_map.items():
                frames = landmarks.shape[0]
                landmarks -= landmark_mean
                landmarks = landmarks.reshape(frames, -1)
                lst.append(landmarks)
        self.__all_landmarks = np.concatenate(lst, axis = 0)

    def __pca_map(self, data, components):
        pca = PCA(n_components = components, copy = False)
        final = pca.fit(data)
        mp = {}
        mp["components"] = copy.deepcopy(final.components_)
        mp["explained_variance"] = copy.deepcopy(final.explained_variance_)
        mp["explained_variance_ratio"] = copy.deepcopy(final.explained_variance_ratio_)
        mp["singular_values"] = copy.deepcopy(final.singular_values_)
        mp["mean"] = copy.deepcopy(final.mean_)
        mp["n_components"] = copy.deepcopy(final.n_components_)
        mp["n_features"] = copy.deepcopy(final.n_features_)
        mp["n_samples"] = copy.deepcopy(final.n_samples_)
        mp["noise_variance"] = copy.deepcopy(final.noise_variance_)
        return mp

    def run(self, components, output_file):
        mp = self.__pca_map(self.__all_landmarks, components)
        with open(output_file, "wb") as fd:
            dump(mp, fd)

def main():
    d = DataPCACalculator()
    d.run(8, "./preprocessed/landmark_pca_8.pkl")

if __name__ == "__main__":
    main()
