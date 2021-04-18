import sys, os, random, torch, pickle
sys.path.append(os.path.dirname(__file__))
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from framework.common_trainer import CommonTrainer
from networks.trainer_interface import MelLandmarkDecoderTrainerInterface
from utils.dataset import ArrayDataset
from tqdm import tqdm
from pathlib import Path
from loguru import logger as log
import matplotlib.pyplot as plt

class LandmarkPredictorTrainer():
    def __init__(self,
        epochs = 100,
        epoch_offset = 1,
        batchsize = 100,
        lr = 0.001,
        landmark_features = 136,
        landmark_mean_path = "./grid_dataset/preprocessed/standard_landmark_mean.pkl",
        landmark_path = "/media/tuantran/raid-data/dataset/GRID/standard_landmark.pkl",
        mel_path = "/media/tuantran/raid-data/dataset/GRID/audio_50/mel.pkl",
        output_path = "./mel_landmark_decoder_output",
        device = "cpu"
    ):
        self.__trainer = CommonTrainer(epochs, epoch_offset, log_interval_second = 30, device = device)
        self.__device = device
        self.__output_path = output_path

        landmark_mean = None
        with open(landmark_mean_path, 'rb') as fd:
            landmark_mean = pickle.load(fd)
        if landmark_mean is None:
            raise Exception("cannot load pca metadata")
        self.__landmark_mean = landmark_mean

        self.__train_dataloader, self.__test_dataloader = self.__create_dataloader(landmark_path, mel_path, batchsize)
        model = MelLandmarkDecoderTrainerInterface(landmark_features, device = device)

        self.__trainer.inject_model(
            model,
        ).inject_optim(
            optim.Adam(model.parameters(), lr = lr)
        ).inject_loss_function(
            torch.nn.MSELoss()
        ).inject_train_dataloader(
            self.__produce_train_data
        ).inject_test_dataloader(
            self.__produce_test_data
        ).inject_evaluation_callback(
            self.__save_evaluation_data
        ).inject_save_model_callback(
            self.__save_model
        )

    def __create_dataloader(self, landmarkpath, melpath, batchsize, training_percentage = 95):
        landmark_data = None
        with open(landmarkpath, 'rb') as fd:
            landmark_data = pickle.load(fd)

        mel_data = None
        with open(melpath, 'rb') as fd:
            mel_data = pickle.load(fd)


        data = []
        for identity, idmap in landmark_data.items():
            for code, lm in idmap.items():
                data.append({
                    'landmarks': lm, 
                    'mel': mel_data[identity][code],
                })

        random.shuffle(data)
        total_data = len(data)
        n_training = int(total_data * (training_percentage/100.0))
        training = data[:n_training]
        testing = data[n_training:]

        def data_processing(item):
            mel = item['mel']
            mel_d = np.diff(mel, n=1, axis=0)
            mel_dd = np.insert(np.diff(mel, n=2, axis=0), 0, np.zeros((1, 64)), axis=0)
            mel = np.concatenate((mel_d, mel_dd), axis = 1)
            mel = torch.tensor(mel).float().unsqueeze(0)

            landmarks = item['landmarks'] - self.__landmark_mean
            frames = landmarks.shape[0]
            landmarks = landmarks.reshape(frames, -1)
            landmarks = torch.tensor(landmarks)

            inspired_landmark = landmarks[3,:]
            landmarks = landmarks[3:-1]

            return ((mel, inspired_landmark), landmarks)

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
        if epoch < 10:
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
        loss = nn.functional.mse_loss(y, yhat)
        log.info(f"mse loss: {loss}")
        yhat = yhat.detach().to("cpu").numpy()
        y = y.detach().to("cpu").numpy()
        for i in range(2):
            fig, axes = plt.subplots(7,10)
            for j in range(70):
                row = j//10
                col = j%10
                ax = axes[row][col]
                predicted_lm = yhat[i][j].reshape(68,2) + self.__landmark_mean
                actual_lm = y[i][j].reshape(68, 2) + self.__landmark_mean
                ax.scatter(actual_lm[:,0], actual_lm[:,1], c = 'black', s = 0.2)
                ax.scatter(predicted_lm[:,0], predicted_lm[:,1], c = 'red', s = 0.2)
                ax.axis('off')
            image_path = os.path.join(folder_path, "landmark_{}.png".format(i))
            fig.set_figheight(7)
            fig.set_figwidth(10)
            plt.savefig(image_path, dpi = 300)
            plt.close()


    def start(self):
        self.__trainer.train()

if __name__ == "__main__":
    trainer = LandmarkPredictorTrainer(device = "cuda:0")
    trainer.start()
