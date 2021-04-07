import sys, os, random, torch, pickle
sys.path.append(os.path.dirname(__file__))
from torch import nn, optim
import torch.nn.functional as F
from compress_pickle import load as cpload
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
from networks.generator import Generator
from networks.discriminator import Discriminator
import time
import numpy as np
from utils.dataset import ArrayDataset
from tqdm import tqdm
from pathlib import Path
from loguru import logger as log
import matplotlib.pyplot as plt

class GansTrainer():
    def __init__(self,
        train_dataloader,
        test_dataloader,
        output_path,
        batchsize,
        adam_lr = 0.0002,
        adam_beta1 = 0.5,
        adam_beta2 = 0.999,
        epochs = 20,
        epoch_offset = 1,
        log_interval_second = 30,
        device = "cpu",
    ):
        self.__device = device
        self.__epochs = epochs
        self.__epoch_offset = epoch_offset
        self.__generator = Generator(device = device)
        self.__discriminator = Discriminator(device = device)
        self.__generator_optim = torch.optim.Adam(
            self.__generator.parameters(),
            lr = adam_lr,
            betas = (adam_beta1, adam_beta2),
        )
        self.__discriminator_optim = torch.optim.Adam(
            self.__discriminator.parameters(),
            lr = adam_lr,
            betas = (adam_beta1, adam_beta2),
        )
        self.__train_dataloader = train_dataloader
        self.__test_dataloader = test_dataloader
        self.__landmark_loss_mask = torch.cat([torch.ones(96), torch.ones(40)*100]).view(1, 1, 136)

        self.__output_path = output_path
        self.__last_log_time = None
        self.__log_interval_second = log_interval_second
        self.__ones_vector = Variable(torch.ones(batchsize), requires_grad = False)
        self.__zeros_vector = Variable(torch.zeros(batchsize), requires_grad = False)

    def __reset_gradients(self):
        self.__generator_optim.zero_grad()
        self.__discriminator_optim.zero_grad()

    def __metric_log(self, epoch, sample, metrics):
        lst = []
        for k, v in metrics.items():
            lst.append("{}: {:.4E}".format(k, v))
        body = ", ".join(lst)
        log.info(f"[epoch {epoch} --- sample {sample}] {body}")

    def __do_logging(self, epoch, sample, metrics):
        now = time.time()
        if now - self.__last_log_time < self.__log_interval_second:
            return
        self.__last_log_time = now
        self.__metric_log(epoch, sample, metrics)

    def start_training(self):
        self.__last_log_time = time.time()
        for epoch in range(self.__epoch_offset, self.__epoch_offset + self.__epochs):
            log.info(f"================================================[epoch {epoch}]================================================")
            for i, ((landmarks, face_images), (inspired_landmark, inspired_image)) in tqdm(enumerate(self.__train_dataloader)):
                landmarks = landmarks.to(self.__device)
                face_images = face_images.to(self.__device)
                inspired_landmark = inspired_landmark.to(self.__device)
                inspired_image = inspired_image.to(self.__device)
                metrics = {}
                """
                start training discriminator
                """
                self.__reset_gradients()
                for p in self.__discriminator.parameters():
                    p.requires_grad =  True

                _, _, fake_images = self.__generator(inspired_image, inspired_landmark, landmarks)
                fake_images = fake_images.detach()

                judgement, judgement_landmarks = self.__discriminator(fake_images, landmarks)
                loss_fake = F.binary_cross_entropy(judgement, self.__zeros_vector)
                loss_fake_landmark = F.mse_loss(
                    judgement_landmarks * self.__landmark_loss_mask,
                    landmarks * self.__landmark_loss_mask
                )

                judgement, judgement_landmarks = self.__discriminator(face_images, landmarks)
                loss_real = F.binary_cross_entropy(judgement, self.__ones_vector)
                loss_real_landmark = F.mse_loss(
                    judgement_landmarks * self.__landmark_loss_mask,
                    landmarks * self.__landmark_loss_mask
                )

                discriminator_loss = loss_fake + loss_real + loss_fake_landmark + loss_real_landmark
                discriminator_loss.backward()
                metrics['discriminator_loss'] = discriminator_loss
                self.__discriminator_optim.step()

                """
                start training generator
                """
                self.__reset_gradients()
                for p in self.__discriminator.parameters():
                    p.requires_grad =  False
                attention_map, color_images, fake_images = self.__generator(inspired_image, inspired_landmark, landmarks)
                judgement, judgement_landmarks = self.__discriminator(fake_images, landmarks)
                loss_generator = F.binary_cross_entropy(judgement, self.__ones_vector)
                loss_generator_landmark = F.mse_loss(
                    judgement_landmarks * self.__landmark_loss_mask,
                    landmarks * self.__landmark_loss_mask
                )
                pixel_loss_mask = attention_map.detach() + 0.5
                pixel_diff = torch.abs(fake_images - face_images) * pixel_loss_mask
                pixel_loss = pixel_diff.mean(pixel_diff)
                final_generator_loss = 10*pixel_loss + loss_generator + loss_generator_landmark
                final_generator_loss.backward()
                metrics['final_generator_loss'] = final_generator_loss
                self.__generator_optim.step()
                self.__reset_gradients()
                
                self.__do_logging(epoch, i, metrics)
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
        self.__output_path = output_path
        self.__train_dataloader, self.__test_dataloader = self.__create_dataloader(landmark_path, face_root_path, batchsize = batchsize)
        self.__trainer = GansTrainer(
            self.__train_dataloader,
            self.__test_dataloader,
            output_path,
            batchsize,
            adam_lr = lr,
            epochs = epochs,
            epoch_offset = epoch_offset,
            device = device,
        )

    def __data_processing(self, item):
        def process_landmark(lm):
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
            face_images = face_images_data['faces'].transpose(0, 2, 3, 1)
            normalized_images = []
            for image in face_images:
                image = transform_ops(image)
                normalized_images.append(image)
            face_images = torch.stack(normalized_images)
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

    def start(self):
        log.info("start training")
        self.__trainer.start_training()

if __name__ == "__main__":
    trainer = FaceGeneratorTrainer(device = "cuda:0")
    trainer.start()
