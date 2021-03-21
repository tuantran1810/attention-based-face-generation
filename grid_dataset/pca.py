import os
import numpy as np
from sklearn.decomposition import PCA
from compress_pickle import load
from pickle import dump

class DataPCACalculator(object):
    def __init__(
        self,
        preprocessed_data = "./preprocessed",
        ext = "gzip",
        output_file = "./preprocessed/grid_pca.pkl"
    ):
        '''
        self.__paths: map[identity]pathsList
        '''

        self.__paths = []

        for path, _ , files in os.walk(preprocessed_data):
            for name in files:
                code, file_ext = name.split('.')
                if file_ext == ext:
                    self.__paths.append(os.path.join(path, name))

        self.__output_file = output_file

    def run(self):
        lst = []
        total = len(self.__paths)
        one_percent = int(total / 100)
        for i, file in enumerate(self.__paths):
            mp = load(file, compression = 'gzip', set_default_extension = False)
            landmarks = mp['landmarks']
            frames = landmarks.shape[0]
            landmarks = landmarks.reshape(frames, -1)
            lst.append(landmarks)
            if (i+1) % one_percent == 0:
                print("{} percent".format((i+1)/one_percent))

        final = np.concatenate(lst, axis = 0)
        print(final.shape)
        pca = PCA(n_components = 6, copy = False)
        final = pca.fit(final)
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
        with open(self.__output_file, "wb") as fd:
            dump(mp, fd)


def main():
    d = DataPCACalculator(preprocessed_data = "/media/tuantran/rapid-data/dataset/GRID/face_images_128")
    d.run()

if __name__ == "__main__":
    main()
