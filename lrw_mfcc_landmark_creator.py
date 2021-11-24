import sys, os, random, torch, pickle
sys.path.append(os.path.dirname(__file__))
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from utils.dataset import ArrayDataset
from tqdm import tqdm
from loguru import logger as log
from networks.landmark_decoder import LandmarkDecoder
import matplotlib.pyplot as plt

class MFCCLandmarkCreator():
    def __init__(self,
        batchsize = 200,
        landmark_features = 8,
        model_path = "./model/lrw/landmark_decoder.pt",
        landmark_pca_path = "./lrw_dataset/preprocessed/landmark_pca_8.pkl",
        landmark_mean_path = "./lrw_dataset/preprocessed/standard_landmark_mean.pkl",
        landmark_path = "/media/tuantran/raid-data/dataset/LRW/attention-based-face-generation/standard_landmark_mobinet.pkl",
        mfcc_path = "/media/tuantran/raid-data/dataset/LRW/attention-based-face-generation/mfcc.pkl",
        output_path = "/media/tuantran/raid-data/dataset/LRW/attention-based-face-generation/generated_pca_landmark_8.pkl",
        device = "cpu",
    ):
        self.__device = device
        self.__output_path = output_path

        landmark_pca_metadata = None
        with open(landmark_pca_path, 'rb') as fd:
            landmark_pca_metadata = pickle.load(fd)
        if landmark_pca_metadata is None:
            raise Exception("cannot load pca metadata")
        self.__landmark_pca_mean = torch.tensor(landmark_pca_metadata["mean"]).unsqueeze(0).to(device)
        self.__landmark_pca_components = torch.tensor(landmark_pca_metadata["components"]).transpose(0, 1).to(device)

        landmark_mean = None
        with open(landmark_mean_path, 'rb') as fd:
            landmark_mean = pickle.load(fd)
        if landmark_mean is None:
            raise Exception("cannot load pca metadata")
        self.__landmark_mean = torch.tensor(landmark_mean).to(device)

        self.__dataloader = self.__create_dataloader(mfcc_path, landmark_path, batchsize)
        self.__model = LandmarkDecoder(landmark_dims = landmark_features, device = device)
        self.__model.load_state_dict(torch.load(model_path))
        # self.__model.eval()

    def __load_data(self, mfcc_path, landmark_path):
        mfcc_data = None
        with open(mfcc_path, 'rb') as fd:
            mfcc_data = pickle.load(fd)

        landmark_data = None
        with open(landmark_path, 'rb') as fd:
            landmark_data = pickle.load(fd)

        result = []

        for utterance, u_map in landmark_data.items():
            for dataset, d_map in u_map.items():
                for code, lm in d_map.items():
                    mfcc = mfcc_data[utterance][dataset][code]
                    result.append(((utterance, dataset, code), (mfcc[:116,1:], lm[3], lm)))
        return result

    def __create_dataloader(self, mfcc_path, landmark_path, batchsize):
        arr = self.__load_data(mfcc_path, landmark_path)

        def data_processing(item):
            r = 3
            window = 23
            (u,d,c), (mfcc, inspired_landmark, orig_landmarks) = item
            start_mfcc = (r - 3) * 4
            end_mfcc = (r + window + 3) * 4
            mfcc = mfcc.transpose(1, 0)[:, start_mfcc:end_mfcc]
            mfcc = torch.tensor(mfcc).float().unsqueeze(0)

            inspired_landmark = torch.tensor(inspired_landmark).float()
            orig_landmarks = torch.tensor(orig_landmarks).float()
            return ((u,d,c), (mfcc, inspired_landmark, orig_landmarks))

        dataset = ArrayDataset(arr, data_processing)
        params = {
            'batch_size': batchsize,
            'shuffle': True,
            'num_workers': 6,
            'drop_last': False,
        }
        return DataLoader(dataset, **params)

    def run(self):
        result = {}
        with torch.no_grad():
            for (u_arr, d_arr, c_arr), (mfcc_arr, inspired_landmark_arr, orig_landmarks) in tqdm(self.__dataloader):
                mfcc_arr = torch.tensor(mfcc_arr).to(self.__device)
                inspired_landmark_arr = torch.tensor(inspired_landmark_arr).to(self.__device)
                batchsize = inspired_landmark_arr.shape[0]
                pca_landmarks = inspired_landmark_arr - self.__landmark_mean
                pca_landmarks = pca_landmarks.reshape(batchsize, -1) - self.__landmark_pca_mean
                pca_landmarks = torch.matmul(pca_landmarks, self.__landmark_pca_components)

                out_landmarks = self.__model(pca_landmarks, mfcc_arr)
                full_landmarks = torch.matmul(out_landmarks, self.__landmark_pca_components.transpose(0,1)) + self.__landmark_pca_mean + self.__landmark_mean.flatten()
                full_landmarks = full_landmarks.detach().to('cpu').numpy()
                # print(full_landmarks.shape)
                out_landmarks = out_landmarks.detach().to('cpu').numpy()

                for i in range(batchsize):
                    lm = out_landmarks[i]
                    u = u_arr[i]
                    d = d_arr[i]
                    c = c_arr[i]
                    if u not in result: result[u] = {}
                    if d not in result[u]: result[u][d] = {}
                    result[u][d][c] = lm
                
                _, axes = plt.subplots(4, 16)
                for i in range(4):
                    for j in range(16):
                        orig = orig_landmarks[i][j]
                        orig = orig.detach().cpu().numpy()
                        lm = full_landmarks[i][j]
                        lm = lm.reshape(68, 2) 
                        axe = axes[i][j]
                        axe.scatter(orig[:,0], orig[:,1], c='b', s=1)
                        axe.scatter(lm[:,0], lm[:,1], c='r', s=1)
                        axe.axis("off")
                        axe.invert_yaxis()
                plt.show()
        # with open(self.__output_path, 'wb') as fd:
        #     pickle.dump(result, fd)


if __name__ == "__main__":
    creator = MFCCLandmarkCreator(
        batchsize = 10,
        device = "cuda:0",
    )
    creator.run()
