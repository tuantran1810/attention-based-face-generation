import sys, os, random, torch, pickle
sys.path.append(os.path.dirname(__file__))
from torch import nn, optim
from compress_pickle import load as cpload
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from framework.gans_trainer import GansTrainer, GansModule
from networks.trainer_interface import GeneratorInterface
from utils.dataset import ArrayDataset
from tqdm import tqdm
from pathlib import Path
from loguru import logger as log
import matplotlib.pyplot as plt

class FaceGeneratorTrainer():
    def __init__(self,
        epochs = 20,
        epoch_offset = 1,
        batchsize = 2,
        lr = 0.0002,
        landmark_path = "/media/tuantran/raid-data/dataset/GRID/attention-based-face-generation/generated_landmark.pkl",
        face_root_path = "/media/tuantran/rapid-data/dataset/GRID/face_images_128",
        output_path = "./face_decoder_output",
        device = "cpu",
    ):
        self.__device = device
        self.__trainer = GansTrainer(epochs = epochs, device = device)
        self.__output_path = output_path
        train_dataloader, test_dataloader = self.__create_dataloader(landmark_path, face_root_path, batchsize = batchsize)

        generator = GeneratorInterface(device = device)
        generator_module = GansModule(
            model = generator,
            optim = optim.Adam(generator.parameters(), lr = lr),
            loss_function = None,
        )

        self.__trainer.inject_train_dataloader(
            train_dataloader,
        ).inject_test_dataloader(
            test_dataloader,
        ).inject_generator(
            generator_module,
        ).inject_discriminator(
            Discriminator(device = device),
        )

    def __data_processing(self, item):
        def process_landmark(lm):
            lm = lm - self.__landmark_mean
            frames = lm.shape[0]
            lm = lm.reshape(frames, -1)
            lm = torch.tensor(lm).float()
            return lm

        landmarks = process_landmark(item['landmarks'])
        inspired_landmark = landmarks[2]
        landmarks = landmarks[3:72]

        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        face_images = None
        with open(item['video_path'], 'rb') as fd:
            face_images_data = cpload(fd, compression = 'gzip')
            face_images = face_images_data['faces']
            face_images = transform_ops(face_images)
        inspired_image = face_images[2]
        face_images = face_images[3:72]

        return ((landmarks, face_images), (inspired_landmark, inspired_image))

    def __create_dataloader(
        self,
        landmark_path,
        root_face_video_path,
        batchsize,
        training_percentage = 95,
        video_file_ext = 'gzip',
    ):
        landmark_data = None
        with open(landmark_path, 'rb') as fd:
            landmark_data = pickle.load(fd)

        face_video_paths = dict()
        for path, _ , files in os.walk(root_face_video_path):
            identity = path.split('/')[-1]
            videomap = {}
            for name in files:
                code, file_ext = name.split('.')
                if file_ext == video_file_ext:
                    videomap[code] = os.path.join(path, name)
            if len(videomap) > 0:
                face_video_paths[identity] = videomap

        data = []
        for identity, idmap in landmark_data.items():
            for code, lm in idmap.items():
                data.append({
                    'landmarks': lm, 
                    'video_path': face_video_paths[identity][code],
                })

        random.shuffle(data)
        total_data = len(data)
        n_training = int(total_data * (training_percentage/100.0))
        training = data[:n_training]
        testing = data[n_training:]

        train_dataset = ArrayDataset(training, self.__data_processing)
        test_dataset = ArrayDataset(testing, self.__data_processing)
        params = {
            'batch_size': batchsize,
            'shuffle': True,
            'num_workers': 6,
            'drop_last': False,
        }
        return DataLoader(train_dataset, **params), DataLoader(test_dataset, **params)

if __name__ == "__main__":
    trainer = FaceGeneratorTrainer(device = "cuda:0")
    trainer.start()
