import sys, os, random, torch, pickle
sys.path.append(os.path.dirname(__file__))
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from framework.common_trainer import CommonTrainer
from networks.trainer_interface import LandmarkDecoderTrainerInterface, LandmarkMSELoss
from utils.dataset import ArrayDataset
from tqdm import tqdm
from pathlib import Path
from loguru import logger as log
import matplotlib.pyplot as plt


class LandmarkPredictorTrainer():
    def __init__(self,
        epochs = 20,
        epoch_offset = 1,
        batchsize = 200,
        lr = 0.0002,
        landmark_pca_dims = 8,
        landmark_pca_path = "./lrw_dataset/preprocessed/landmark_pca_8.pkl",
        landmark_mean_path = "./lrw_dataset/preprocessed/standard_landmark_mean.pkl",
        landmark_path = "/media/tuantran/raid-data/dataset/LRW/attention-based-face-generation/standard_landmark_mobinet.pkl",
        mfcc_path = "/media/tuantran/raid-data/dataset/LRW/attention-based-face-generation/mfcc.pkl",
        output_path = "./landmark_decoder_output",
        device = "cpu"
    ):
        self.__trainer = CommonTrainer(epochs, epoch_offset, log_interval_second = 10, device = device)
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

        self.__landmark_pca_mean = torch.tensor(landmark_pca_metadata["mean"]).to(device)
        self.__landmark_pca_components = torch.tensor(landmark_pca_metadata["components"]).to(device)

        self.__train_dataloader, self.__test_dataloader = self.__create_dataloader(landmark_path, mfcc_path, batchsize)
        model = LandmarkDecoderTrainerInterface(landmark_pca_dims, self.__landmark_pca_mean, self.__landmark_pca_components, device = device)

        self.__trainer.inject_model(
            model,
        ).inject_optim(
            optim.Adam(model.parameters(), lr = lr)
        ).inject_loss_function(
            LandmarkMSELoss(self.__landmark_pca_mean, self.__landmark_pca_components.transpose(0, 1), y_padding = 3, device = device)
        ).inject_train_dataloader(
            self.__produce_train_data
        ).inject_test_dataloader(
            self.__produce_test_data
        ).inject_evaluation_callback(
            self.__save_evaluation_data
        ).inject_save_model_callback(
            self.__save_model
        )

    def __load_landmarks(self, landmark_path, which_data):
        returned_data = dict()
        with open(landmark_path, 'rb') as fd:
            landmark_map = pickle.load(fd)
            for utterance in landmark_map:
                if which_data not in landmark_map[utterance]:
                    continue
                dataset = landmark_map[utterance][which_data]
                for code, landmarks in dataset.items():
                    returned_data[code] = landmarks
        return returned_data

    def __load_mfcc(self, mfcc_path, which_data):
        returned_data = dict()
        data = None
        with open(mfcc_path, 'rb') as fd:
            data = pickle.load(fd)
        for utterance in data:
            if which_data not in data[utterance]:
                continue
            dataset = data[utterance][which_data]
            for code in dataset:
                if dataset[code] is None:
                    print(f"none record for code {code}")
                    continue
                if code in returned_data and returned_data[code] is not None:
                    print(f"code {code} existed in data")
                    continue
                returned_data[code] = dataset[code]
        return returned_data


    def __create_dataloader(self, landmark_path, mfcc_path, batchsize):
        landmark_train = self.__load_landmarks(landmark_path, 'train')
        landmark_val = self.__load_landmarks(landmark_path, 'val')
        mfcc_train = self.__load_mfcc(mfcc_path, 'train')
        mfcc_val = self.__load_mfcc(mfcc_path, 'val')

        data_train = []
        for code, lm in landmark_train.items():
            if code not in mfcc_train:
                continue
            data_train.append({
                'landmarks': lm, 
                'mfcc': mfcc_train[code][:116,1:],
            })

        data_val = []
        for code, lm in landmark_val.items():
            if code not in mfcc_val:
                continue
            data_val.append({
                'landmarks': lm, 
                'mfcc': mfcc_val[code][:116,1:],
            })

        def data_processing(item):
            r = 3
            window = 23
            mfcc = item['mfcc']
            start_mfcc = (r - 3) * 4
            end_mfcc = (r + window + 3) * 4
            mfcc = mfcc.transpose(1, 0)[:, start_mfcc:end_mfcc]
            mfcc = torch.tensor(mfcc).float().unsqueeze(0)

            landmarks = item['landmarks']
            landmarks = landmarks - self.__landmark_mean
            frames = landmarks.shape[0]
            landmarks = landmarks.reshape(frames, -1)
            landmarks = torch.tensor(landmarks).float()

            inspired_landmark = landmarks[r,:]
            landmarks = landmarks[r+1:r+1+window]
            return ((mfcc, inspired_landmark), landmarks)

        train_dataset = ArrayDataset(data_train, data_processing)
        val_dataset = ArrayDataset(data_val, data_processing)
        params = {
            'batch_size': batchsize,
            'shuffle': True,
            'num_workers': 6,
            'drop_last': False,
        }
        return DataLoader(train_dataset, **params), DataLoader(val_dataset, **params)

    def __produce_train_data(self):
        loader = tqdm(self.__train_dataloader, desc = "Training")
        for ((mfcc, inspired_landmark), landmarks) in loader:
            mfcc = mfcc.to(self.__device)
            inspired_landmark = inspired_landmark.to(self.__device)
            landmarks = landmarks.to(self.__device)
            yield ((mfcc, inspired_landmark), landmarks)

    def __produce_test_data(self):
        loader = tqdm(self.__test_dataloader, desc = "Testing")
        for ((mfcc, inspired_landmark), landmarks) in loader:
            mfcc = mfcc.to(self.__device)
            inspired_landmark = inspired_landmark.to(self.__device)
            landmarks = landmarks.to(self.__device)
            yield ((mfcc, inspired_landmark), landmarks)

    def __save_model(self, epoch, model):
        if epoch < 3:
            return
        log.info(f"saving model for epoch {epoch}")
        models_folder = "models"
        epoch_folder = "epoch_{}".format(epoch)
        folder_path = os.path.join(self.__output_path, models_folder, epoch_folder)
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        model_path = os.path.join(folder_path, "model.pt")
        torch.save(model.state_dict(), model_path)

    def __save_evaluation_data(self, epoch, sample, x, y, yhat):
        if sample != 0:
            return
        data_folder = "data"
        epoch_folder = "epoch_{}".format(epoch)
        folder_path = os.path.join(self.__output_path, data_folder, epoch_folder)
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        for i in range(10):
            fig, axes = plt.subplots(3,8)
            for j in range(23):
                row = j//8
                col = j%8
                ax = axes[row][col]
                predicted_lm = torch.matmul(yhat[i][j], self.__landmark_pca_components) + self.__landmark_pca_mean
                predicted_lm = predicted_lm.detach().to("cpu").numpy().reshape(68,2) + self.__landmark_mean
                actual_lm = y[i][j].reshape(68, 2).detach().to("cpu").numpy() + self.__landmark_mean
                ax.scatter(actual_lm[:,0], actual_lm[:,1], c = 'black', s = 0.2)
                ax.scatter(predicted_lm[:,0], predicted_lm[:,1], c = 'red', s = 0.2)
                ax.axis('off')
            image_path = os.path.join(folder_path, "landmark_{}.png".format(i))
            fig.set_figheight(3)
            fig.set_figwidth(8)
            plt.savefig(image_path, dpi = 500)
            plt.close()

    def start(self):
        self.__trainer.train()

if __name__ == "__main__":
    trainer = LandmarkPredictorTrainer(device = "cuda:0")
    trainer.start()
