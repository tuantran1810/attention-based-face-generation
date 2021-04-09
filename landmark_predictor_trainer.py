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

'''
This training produce the test loss = 4.92e-4 at the 17th epoch
'''

class LandmarkPredictorTrainer():
    def __init__(self,
        epochs = 20,
        epoch_offset = 1,
        batchsize = 200,
        lr = 0.0002,
        landmark_features = 7,
        landmark_pca_path = "./grid_dataset/preprocessed/landmark_pca.pkl",
        landmark_mean_path = "./grid_dataset/preprocessed/landmark_mean.pkl",
        landmark_path = "/media/tuantran/raid-data/dataset/GRID/standard_landmark.pkl",
        mfcc_path = "/media/tuantran/raid-data/dataset/GRID/audio_50/mfcc.pkl",
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

        landmark_pca_metadata = landmark_pca_metadata[landmark_features]
        self.__landmark_pca_mean = torch.tensor(landmark_pca_metadata["mean"]).to(device)
        self.__landmark_pca_components = torch.tensor(landmark_pca_metadata["components"]).to(device)

        self.__train_dataloader, self.__test_dataloader = self.__create_dataloader(landmark_path, mfcc_path, batchsize)
        model = LandmarkDecoderTrainerInterface(landmark_features, self.__landmark_pca_mean, self.__landmark_pca_components, device = device)

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

    def __create_dataloader(self, landmarkpath, mfccpath, batchsize, training_percentage = 95):
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
                    'landmarks': lm, 
                    'mfcc': mfcc_data[identity][code],
                })

        random.shuffle(data)
        total_data = len(data)
        n_training = int(total_data * (training_percentage/100.0))
        training = data[:n_training]
        testing = data[n_training:]

        def data_processing(item):
            r = random.choice([x for x in range(6,50)])
            window = 16
            mfcc = item['mfcc']
            start_mfcc = (r - 3) * 4
            end_mfcc = (r + window + 3) * 4
            mfcc = mfcc.transpose(1, 0)[1:, start_mfcc:end_mfcc]
            mfcc = torch.tensor(mfcc).float().unsqueeze(0)

            landmarks = item['landmarks']
            landmarks = landmarks - self.__landmark_mean
            frames = landmarks.shape[0]
            landmarks = landmarks.reshape(frames, -1)
            landmarks = torch.tensor(landmarks).float()

            inspired_landmark = landmarks[r,:]
            landmarks = landmarks[r+1:r+1+window]
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
        if epoch < 7:
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
        for i in range(2):
            fig, axes = plt.subplots(4,4)
            for j in range(16):
                row = j//4
                col = j%4
                ax = axes[row][col]
                predicted_lm = torch.matmul(yhat[i][j], self.__landmark_pca_components) + self.__landmark_pca_mean
                predicted_lm = predicted_lm.detach().to("cpu").numpy().reshape(68,2) + self.__landmark_mean
                actual_lm = y[i][j].reshape(68, 2).detach().to("cpu").numpy() + self.__landmark_mean
                ax.scatter(actual_lm[:,0], actual_lm[:,1], c = 'black', s = 0.2)
                ax.scatter(predicted_lm[:,0], predicted_lm[:,1], c = 'red', s = 0.2)
                ax.axis('off')
            image_path = os.path.join(folder_path, "landmark_{}.png".format(i))
            fig.set_figheight(5)
            fig.set_figwidth(5)
            plt.savefig(image_path, dpi = 500)
            plt.close()

    def start(self):
        self.__trainer.train()

if __name__ == "__main__":
    trainer = LandmarkPredictorTrainer(device = "cuda:0")
    trainer.start()
