import sys, os
sys.path.append(os.path.dirname(__file__))
import random
import torch
import pickle
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
        epochs = 10,
        epoch_offset = 1,
        batchsize = 150,
        lr = 0.0001,
        landmark_features = 6,
        landmark_pca_path = "./grid_dataset/preprocessed/grid_pca.pkl",
        mfcc_norm_path = "./grid_dataset/preprocessed/grid_mfcc_norm.pkl",
        data_path = "/media/tuantran/rapid-data/dataset/GRID/face_images_128/mfcc_landmark.pkl",
        output_path = "./landmark_decoder_output",
        device = "cpu"
    ):
        self.__trainer = CommonTrainer(epochs, epoch_offset, device = device)
        self.__device = device
        self.__output_path = output_path
        
        landmark_pca_metadata = None
        with open(landmark_pca_path, 'rb') as fd:
            landmark_pca_metadata = pickle.load(fd)
        if landmark_pca_metadata is None:
            raise Exception("cannot load pca metadata")
        landmark_pca_metadata = landmark_pca_metadata[landmark_features]
        self.__landmark_pca_mean = torch.tensor(landmark_pca_metadata["mean"]).to(device)
        self.__landmark_pca_components = torch.tensor(landmark_pca_metadata["components"]).to(device)

        mp = None
        with open(mfcc_norm_path, 'rb') as fd:
            mp = pickle.load(fd)
        if mp is None:
            raise Exception("cannot load pca metadata")
        self.__mfcc_max_value = mp['max']
        self.__mfcc_min_value = mp['min']

        self.__train_dataloader, self.__test_dataloader = self.__create_dataloader(data_path, batchsize)
        model = LandmarkDecoderTrainerInterface(landmark_features, self.__landmark_pca_mean, self.__landmark_pca_components, device = device)
        loss_weight = torch.ones(6)

        self.__trainer.inject_model(
            model,
        ).inject_optim(
            optim.Adam(model.parameters(), lr = lr)
        ).inject_loss_function(
            LandmarkMSELoss(self.__landmark_pca_mean, self.__landmark_pca_components.transpose(0, 1), weight = loss_weight, y_padding = 3, device = device)
        ).inject_train_dataloader(
            self.__produce_train_data
        ).inject_test_dataloader(
            self.__produce_test_data
        ).inject_evaluation_callback(
            self.__save_evaluation_data
        ).inject_save_model_callback(
            self.__save_model
        )
    def __create_dataloader(self, datapath, batchsize, training_percentage = 95):
        data = None
        with open(datapath, 'rb') as fd:
            data = pickle.load(fd)
        if data is None:
            raise Exception("cannot load training data")

        random.shuffle(data)
        total_data = len(data)
        n_training = int(total_data * (training_percentage/100.0))
        training = data[:n_training]
        testing = data[n_training:]

        def data_processing(item):
            mfcc = item['mfcc']
            mfcc = (mfcc - self.__mfcc_min_value) / (self.__mfcc_max_value - self.__mfcc_min_value)
            mfcc = torch.tensor(mfcc).float()
            f = mfcc.shape[1]
            mfcc = mfcc.transpose(0, 1).reshape(f, -1).unsqueeze(0)

            landmarks = item['landmarks']
            frames = landmarks.shape[0]
            landmarks = landmarks.reshape(frames, -1)
            landmarks = torch.tensor(landmarks)

            inspired_landmark = landmarks[0,:]

            return ((mfcc, inspired_landmark), landmarks)

        train_dataset = ArrayDataset(training, data_processing)
        test_dataset = ArrayDataset(testing, data_processing)
        params = {
            'batch_size': batchsize,
            'shuffle': True,
            'num_workers': 6,
            'drop_last': False,
        }
        return DataLoader(train_dataset, **params), DataLoader(test_dataset, **params)

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
        for i in range(2):
            fig, axes = plt.subplots(7, 10)
            for j in range(69):
                row = j//10
                col = j%10
                ax = axes[row][col]
                predicted_lm = torch.matmul(yhat[i][j], self.__landmark_pca_components) + self.__landmark_pca_mean
                predicted_lm = predicted_lm.detach().to("cpu").numpy().reshape(68,2)
                actual_lm = y[i][j].reshape(68, 2).detach().to("cpu").numpy()
                ax.scatter(actual_lm[:,0], actual_lm[:,1], c = 'black', s = 0.2)
                ax.scatter(predicted_lm[:,0], predicted_lm[:,1], c = 'red', s = 0.2)
                ax.axis('off')
            image_path = os.path.join(folder_path, "landmark_{}.png".format(i))
            fig.set_figheight(7)
            fig.set_figwidth(10)
            plt.savefig(image_path, dpi = 500)

    def start(self):
        self.__trainer.train()

if __name__ == "__main__":
    trainer = LandmarkPredictorTrainer(device = "cuda:0")
    trainer.start()
