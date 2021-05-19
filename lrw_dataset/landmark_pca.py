import numpy as np
from sklearn.decomposition import PCA
from pickle import dump, load
from tqdm import tqdm
import copy

class DataPCACalculator(object):
    def __init__(
        self,
        metadata_list = ["./preprocessed/list_0.pkl", "./preprocessed/list_1.pkl"],
        landmark_mean_path = "./preprocessed/standard_landmark_mean.pkl",
    ):
        landmark_mean = None
        with open(landmark_mean_path, 'rb') as fd:
            landmark_mean = load(fd)
        if landmark_mean is None:
            raise Exception("cannot load pca metadata")
        landmark_mean = landmark_mean.astype(np.float32)

        lst = []
        for metadata_file in metadata_list:
            metadata = None
            with open(metadata_file, 'rb') as fd:
                metadata = load(fd)
            print(f"loading: {metadata_file}")
            for utterance, utterance_map in metadata.items():
                if 'train' not in utterance_map:
                    print(f"there is no train in {utterance}")
                    continue
                code_map = utterance_map['train']
                for _, video_metadata in code_map.items():
                    video_path = video_metadata['path']
                    ltrb = video_metadata['ltrb']
                    landmark = video_metadata['landmark']
                    if video_path is None or ltrb is None or landmark is None or len(ltrb) != 4 or landmark.shape != (29,68,2):
                        continue
                    landmark -= landmark_mean
                    frames = landmark.shape[0]
                    landmark = landmark.reshape(frames, -1)
                    lst.append(landmark)
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
    paths = []
    for i in range(6):
        paths.append(f"/media/tuantran/raid-data/dataset/LRW/attention-based-face-generation/list_data/list_{i}.pkl")

    d = DataPCACalculator(
        metadata_list = paths,
        landmark_mean_path = "./preprocessed/standard_landmark_mean.pkl",
    )
    d.run(30, "./preprocessed/landmark_pca_30.pkl")

if __name__ == "__main__":
    main()
