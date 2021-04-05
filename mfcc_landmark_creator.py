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
        landmark_features = 7,
        model_path = "./model/landmark_decoder.pt",
        landmark_pca_path = "./grid_dataset/preprocessed/landmark_pca.pkl",
        landmark_mean_path = "./grid_dataset/preprocessed/landmark_mean.pkl",
        landmark_path = "/media/tuantran/raid-data/dataset/GRID/standard_landmark.pkl",
        mfcc_path = "/media/tuantran/raid-data/dataset/GRID/audio_50/mfcc.pkl",
        output_path = "/media/tuantran/raid-data/dataset/GRID/attention-based-face-generation/generated_landmark.pkl",
        device = "cpu"
    ):
        self.__device = device
        self.__output_path = output_path
        
        landmark_pca_metadata = None
        with open(landmark_pca_path, 'rb') as fd:
            landmark_pca_metadata = pickle.load(fd)
        if landmark_pca_metadata is None:
            raise Exception("cannot load pca metadata")

        landmark_mean = None
        with open(landmark_mean_path, 'rb') as fd:
            landmark_mean = pickle.load(fd)
        if landmark_mean is None:
            raise Exception("cannot load pca metadata")
        self.__landmark_mean = landmark_mean

        landmark_pca_metadata = landmark_pca_metadata[landmark_features]
        self.__landmark_pca_mean = torch.tensor(landmark_pca_metadata["mean"]).unsqueeze(0).to(device)
        self.__landmark_pca_components = torch.tensor(landmark_pca_metadata["components"]).to(device)

        self.__dataloader = self.__create_dataloader(landmark_path, mfcc_path, batchsize)
        self.__model = LandmarkDecoder(landmark_dims = landmark_features, device = device)
        self.__model.load_state_dict(torch.load(model_path))
        self.__model.eval()

    def __create_dataloader(self, landmarkpath, mfccpath, batchsize):
        landmark_data = None
        with open(landmarkpath, 'rb') as fd:
            landmark_data = pickle.load(fd)

        mfcc_data = None
        with open(mfccpath, 'rb') as fd:
            mfcc_data = pickle.load(fd)


        data = []
        for identity, idmap in landmark_data.items():
            for code, lm in idmap.items():
                data.append({
                    'identity': identity,
                    'code': code,
                    'landmarks': lm, 
                    'mfcc': mfcc_data[identity][code],
                })

        random.shuffle(data)

        def data_processing(item):
            mfcc = item['mfcc']
            mfcc = mfcc.transpose(1, 0)[1:,:]
            mfcc = torch.tensor(mfcc).float().unsqueeze(0)

            landmarks = item['landmarks']
            landmarks = landmarks - self.__landmark_mean
            frames = landmarks.shape[0]
            landmarks = landmarks.reshape(frames, -1)
            landmarks = torch.tensor(landmarks).float()

            inspired_landmark = landmarks[2,:]
            landmarks = landmarks[3:72]
            return ((item['identity'], item['code']), (mfcc, inspired_landmark), landmarks)

        dataset = ArrayDataset(data, data_processing)
        params = {
            'batch_size': batchsize,
            'shuffle': True,
            'num_workers': 6,
            'drop_last': False,
        }
        return DataLoader(dataset, **params)

    def __produce_data(self):
        loader = tqdm(self.__dataloader, desc = "Producing")
        for ((identity, code), (mfcc, inspired_landmark), landmarks) in loader:
            mfcc = mfcc.to(self.__device)
            inspired_landmark = inspired_landmark.to(self.__device)
            landmarks = landmarks.to(self.__device)
            yield ((identity, code), (mfcc, inspired_landmark), landmarks)

    def start(self):
        data = {}
        with torch.no_grad():
            for (identities, codes), (mfcc, inspired_landmark), _ in self.__produce_data():
                pca_landmarks = torch.matmul(inspired_landmark - self.__landmark_pca_mean, self.__landmark_pca_components.transpose(0,1))
                yhat = self.__model(pca_landmarks, mfcc)
                yhat = torch.matmul(yhat, self.__landmark_pca_components) + self.__landmark_pca_mean
                
                batchsize = yhat.shape[0]
                for i in range(batchsize):
                    identity, code = identities[i], codes[i]
                    yhati = yhat[i]
                    frames = yhati.shape[0]
                    yhati = yhati.view(frames, 68, -1)
                    if identity not in data:
                        data[identity] = dict()
                    data[identity][code] = yhati.detach().to('cpu').numpy()
        
        with open(self.__output_path, 'wb') as fd:
            pickle.dump(data, fd)

if __name__ == "__main__":
    trainer = MFCCLandmarkCreator(device = "cuda:0")
    trainer.start()
