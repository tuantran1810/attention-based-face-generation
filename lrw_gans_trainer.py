import sys, os, random, torch, pickle, cv2
sys.path.append(os.path.dirname(__file__))
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
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
        pretrained_model_paths = dict(),
        log_interval_second = 30,
        device = "cpu",
    ):
        self.__device = device
        self.__epochs = epochs
        self.__epoch_offset = epoch_offset
        self.__generator = Generator(device = device)
        self.__discriminator = Discriminator(device = device)
        if 'generator' in pretrained_model_paths:
            path = pretrained_model_paths['generator']
            log.info(f"reload generator from {path}")
            self.__generator.load_state_dict(torch.load(path))

        if 'discriminator' in pretrained_model_paths:
            path = pretrained_model_paths['discriminator']
            log.info(f"reload discriminator from {path}")
            self.__discriminator.load_state_dict(torch.load(path))

        self.__generator_optim = torch.optim.Adam(
            self.__generator.parameters(),
            lr = adam_lr,
            betas = (adam_beta1, adam_beta2),
        )
        self.__grad_scaler_generator = GradScaler()

        self.__discriminator_optim = torch.optim.Adam(
            self.__discriminator.parameters(),
            lr = adam_lr,
            betas = (adam_beta1, adam_beta2),
        )
        self.__grad_scaler_discriminator = GradScaler()
        self.__train_dataloader = train_dataloader
        self.__test_dataloader = test_dataloader
        self.__landmark_loss_mask = torch.cat([torch.ones(96), torch.ones(40)*100]).view(1, 1, 136).to(device)

        self.__output_path = output_path
        self.__last_log_time = None
        self.__log_interval_second = log_interval_second
        self.__ones_vector = Variable(torch.ones(batchsize), requires_grad = False).to(device)
        self.__zeros_vector = Variable(torch.zeros(batchsize), requires_grad = False).to(device)

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

    def __save_evaluation_data(self,
        epoch,
        sample,
        utterance,
        dataset,
        code,
        original_images,
        attention_map,
        color_images,
        fake_images,
    ):
        if sample > 0:
            return
        data_folder = "data"
        folder_path = os.path.join(self.__output_path, data_folder)
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        original_images = np.uint8((original_images.cpu().numpy() + 1.0) * 127.5)
        attention_map = attention_map.cpu().numpy()
        color_images = np.uint8((color_images.cpu().numpy() + 1.0) * 127.5)
        fake_images = np.uint8((fake_images.cpu().numpy() + 1.0) * 127.5)
        data = {
            'utterance': utterance,
            'code': code,
            'dataset': dataset,
            'original_images': original_images,
            'attention_map': attention_map,
            'color_images': color_images,
            'fake_images': fake_images,
        }
        data_path = os.path.join(folder_path, "data_{}.pkl".format(epoch))
        with open(data_path, 'wb') as fd:
            pickle.dump(data, fd)

    def __save_model(self, epoch):
        log.info(f"saving model for epoch {epoch}")
        models_folder = "models"
        epoch_folder = "epoch_{}".format(epoch)
        folder_path = os.path.join(self.__output_path, models_folder, epoch_folder)
        Path(folder_path).mkdir(parents=True, exist_ok=True)

        generator_path = os.path.join(folder_path, "generator.pt")
        torch.save(self.__generator.state_dict(), generator_path)
        discrinminator_path = os.path.join(folder_path, 'discriminator.pt')
        torch.save(self.__discriminator.state_dict(), discrinminator_path)

    def start_training(self):
        self.__last_log_time = time.time()
        for epoch in range(self.__epoch_offset, self.__epoch_offset + self.__epochs):
            log.info(f"================================================[epoch {epoch}]================================================")
            i = 0
            log.info("start training...")
            for item in tqdm(self.__train_dataloader, "training"):
                i += 1
                (u, d, c), (gen_landmarks, orig_landmarks, face_images), (inspired_landmark, inspired_image) = item
                gen_landmarks = gen_landmarks.to(self.__device)
                orig_landmarks = orig_landmarks.to(self.__device)
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

                with autocast():
                    _, _, fake_images = self.__generator(inspired_image, inspired_landmark, gen_landmarks)
                    fake_images = fake_images.detach()
                    judgement, judgement_landmarks = self.__discriminator(fake_images, inspired_landmark)
                    loss_fake = F.binary_cross_entropy_with_logits(judgement, self.__zeros_vector)

                    masked_judgement_landmarks = judgement_landmarks * self.__landmark_loss_mask
                    masked_landmarks = orig_landmarks * self.__landmark_loss_mask
                    loss_fake_landmark = F.mse_loss(
                        masked_judgement_landmarks,
                        masked_landmarks
                    )

                    judgement, judgement_landmarks = self.__discriminator(face_images, inspired_landmark)
                    masked_judgement_landmarks = judgement_landmarks * self.__landmark_loss_mask
                    masked_landmarks = orig_landmarks * self.__landmark_loss_mask
                    loss_real = F.binary_cross_entropy_with_logits(judgement, self.__ones_vector)
                    loss_real_landmark = F.mse_loss(
                        masked_judgement_landmarks,
                        masked_landmarks
                    )

                    discriminator_loss = loss_fake + loss_real + loss_fake_landmark + loss_real_landmark

                self.__grad_scaler_discriminator.scale(discriminator_loss).backward()
                self.__grad_scaler_discriminator.step(self.__discriminator_optim)
                self.__grad_scaler_discriminator.update()
                metrics['discriminator_loss'] = discriminator_loss

                """
                start training generator
                """
                self.__reset_gradients()
                for p in self.__discriminator.parameters():
                    p.requires_grad =  False

                with autocast():
                    attention_map, color_images, fake_images = self.__generator(inspired_image, inspired_landmark, gen_landmarks)
                    judgement, judgement_landmarks = self.__discriminator(fake_images, inspired_landmark)
                    loss_generator = F.binary_cross_entropy_with_logits(judgement, self.__ones_vector)
                    loss_generator_landmark = F.mse_loss(
                        judgement_landmarks * self.__landmark_loss_mask,
                        orig_landmarks * self.__landmark_loss_mask
                    )
                    pixel_loss_mask = attention_map.detach() + 0.5
                    pixel_diff = fake_images - face_images
                    pixel_diff = torch.abs(pixel_diff) * pixel_loss_mask
                    pixel_loss = torch.mean(pixel_diff)
                    final_generator_loss = 10*pixel_loss + loss_generator + loss_generator_landmark

                self.__grad_scaler_generator.scale(final_generator_loss).backward()
                metrics['pixel_loss'] = pixel_loss
                metrics['loss_generator'] = loss_generator
                metrics['loss_generator_landmark'] = loss_generator_landmark
                metrics['final_generator_loss'] = final_generator_loss

                self.__grad_scaler_generator.step(self.__generator_optim)
                self.__grad_scaler_generator.update()
                self.__reset_gradients()

                self.__do_logging(epoch, i, metrics)

            log.info("start evaluating...")
            with torch.no_grad():
                cnt = 0.0
                pixel_loss_arr = []
                loss_generator_arr = []
                loss_generator_landmark_arr = []
                final_generator_loss_arr = []
                i = 0
                for item in tqdm(self.__test_dataloader, "evaluating"):
                    cnt += 1.0
                    (u, d, c), (gen_landmarks, orig_landmarks, face_images), (inspired_landmark, inspired_image) = item
                    gen_landmarks = gen_landmarks.to(self.__device)
                    orig_landmarks = orig_landmarks.to(self.__device)
                    face_images = face_images.to(self.__device)
                    inspired_landmark = inspired_landmark.to(self.__device)
                    inspired_image = inspired_image.to(self.__device)

                    attention_map, color_images, fake_images = self.__generator(inspired_image, inspired_landmark, gen_landmarks)
                    judgement, judgement_landmarks = self.__discriminator(fake_images, inspired_landmark)
                    loss_generator = F.binary_cross_entropy_with_logits(judgement, self.__ones_vector)
                    loss_generator_landmark = F.mse_loss(
                        judgement_landmarks * self.__landmark_loss_mask,
                        orig_landmarks * self.__landmark_loss_mask,
                    )
                    pixel_loss_mask = attention_map.detach() + 0.5
                    pixel_diff = fake_images - face_images
                    pixel_diff = torch.abs(pixel_diff) * pixel_loss_mask
                    pixel_loss = torch.mean(pixel_diff)
                    final_generator_loss = 10*pixel_loss + loss_generator + loss_generator_landmark

                    pixel_loss_arr.append(pixel_loss)
                    loss_generator_arr.append(loss_generator)
                    loss_generator_landmark_arr.append(loss_generator_landmark)
                    final_generator_loss_arr.append(final_generator_loss)
                    self.__save_evaluation_data(
                        epoch,
                        i,
                        u,
                        d,
                        c,
                        face_images,
                        attention_map,
                        color_images,
                        fake_images
                    )
                    i += 1

                metrics = {
                    "pixel_loss": sum(pixel_loss_arr)/cnt,
                    'loss_generator': sum(loss_generator_arr)/cnt,
                    'loss_generator_landmark': sum(loss_generator_landmark_arr)/cnt,
                    'final_generator_loss': sum(final_generator_loss_arr)/cnt,
                }
                self.__metric_log(epoch, -1, metrics)
            self.__save_model(epoch)

class FaceGeneratorTrainer():
    def __init__(self,
        epochs = 20,
        epoch_offset = 1,
        batchsize = 6,
        lr = 0.0002,
        landmark_pca_path = "./lrw_dataset/preprocessed/landmark_pca_8.pkl",
        landmark_mean_path = "./lrw_dataset/preprocessed/standard_landmark_mean.pkl",
        generated_pca_landmark_path = "/media/tuantran/rapid-data/dataset/LRW/attention-based-face-generation/generated_pca_landmark_8.pkl",
        original_landmark_path = "/media/tuantran/rapid-data/dataset/LRW/attention-based-face-generation/raw_landmark.pkl",
        face_root_path = "/media/tuantran/rapid-data/dataset/LRW/attention-based-face-generation/training_video",
        output_path = "./face_decoder_output",
        pretrained_model_paths = dict(),
        device = "cpu",
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
        self.__landmark_mean = landmark_mean.reshape(1, -1)

        self.__landmark_pca_mean = np.expand_dims(landmark_pca_metadata["mean"], 0)
        self.__landmark_pca_components = landmark_pca_metadata["components"]

        train_dataloader, test_dataloader = self.__create_dataloader(generated_pca_landmark_path, original_landmark_path, face_root_path, batchsize = batchsize)
        self.__trainer = GansTrainer(
            train_dataloader,
            test_dataloader,
            output_path,
            batchsize,
            adam_lr = lr,
            epochs = epochs,
            epoch_offset = epoch_offset,
            pretrained_model_paths = pretrained_model_paths,
            device = device,
        )

    def __data_processing(self, item):
        def process_pca_landmark(pca_landmarks):
            landmark = np.matmul(pca_landmarks, self.__landmark_pca_components) + self.__landmark_pca_mean
            landmark = landmark + self.__landmark_mean - 0.5

            # frames = landmark.shape[0]
            # showlm = landmark.reshape(frames, 68, 2)
            # _, axes = plt.subplots(4, 6)
            # for i in range(23):
            #     r, c = i//6, i%6
            #     lm = showlm[i]
            #     axe = axes[r][c]
            #     axe.scatter(lm[:,0], lm[:,1])

            landmark = torch.tensor(landmark).float()
            return landmark

        def process_inspired_landmark(landmark):
            landmark = landmark.flatten()
            landmark = landmark - 0.5
            landmark = torch.tensor(landmark).float()
            return landmark

        def process_original_landmark(landmarks):
            frames = landmarks.shape[0]
            landmarks = landmarks.reshape(frames, -1)

            # frames = landmarks.shape[0]
            # showlm = landmarks.reshape(frames, 68, 2)
            # _, axes = plt.subplots(4, 6)
            # for i in range(23):
            #     r, c = i//6, i%6
            #     lm = showlm[i]
            #     axe = axes[r][c]
            #     axe.scatter(lm[:,0], lm[:,1])

            landmarks = landmarks - 0.5
            return torch.tensor(landmarks).float()

        def iterate_frames(videofile):
            vidcap = cv2.VideoCapture(videofile)
            while True:
                success, image = vidcap.read()
                if not success:
                    return
                if image is None:
                    print("image is None")
                yield image

        generated_landmarks = process_pca_landmark(item['glm'])
        inspired_landmark = process_inspired_landmark(item['ilm'])
        original_landmarks = process_original_landmark(item['olm'])
        # print(item['udc'])
        # plt.show()
        # plt.close()

        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        normalized_images = []
        for image in iterate_frames(item['vpath']):
            image = transform_ops(image)
            normalized_images.append(image)
        face_images = torch.stack(normalized_images)

        inspired_image = face_images[3]
        face_images = face_images[4:27].permute(1,0,2,3)
        original_landmarks = generated_landmarks
        return (item['udc'], (generated_landmarks, original_landmarks, face_images), (inspired_landmark, inspired_image))

    def __create_dataloader(
        self,
        generated_pca_landmark_path,
        original_landmark_path,
        root_face_video_path,
        batchsize,
    ):
        generated_landmark_data = None
        with open(generated_pca_landmark_path, 'rb') as fd:
            generated_landmark_data = pickle.load(fd)

        original_landmarks = {}
        with open(original_landmark_path, 'rb') as fd:
            original_landmarks = pickle.load(fd)

        train_data = []
        dataset = 'train'
        for u, umap in generated_landmark_data.items():
            dmap = umap[dataset]
            for c, lm in dmap.items():
                olm = original_landmarks[u][dataset][c]
                train_data.append({
                    'udc': (u, dataset, c),
                    'vpath': os.path.join(root_face_video_path, u, dataset, c+'.mp4'),
                    'glm': lm,
                    'olm': olm[4:27],
                    'ilm': olm[3], 
                })

        test_data = []
        dataset = 'val'
        for u, umap in generated_landmark_data.items():
            dmap = umap[dataset]
            for c, lm in dmap.items():
                olm = original_landmarks[u][dataset][c]
                test_data.append({
                    'udc': (u, dataset, c),
                    'vpath': os.path.join(root_face_video_path, u, dataset, c+'.mp4'),
                    'glm': lm, 
                    'olm': olm[4:27], 
                    'ilm': olm[3], 
                })

        train_dataset = ArrayDataset(train_data, self.__data_processing)
        test_dataset = ArrayDataset(test_data, self.__data_processing)
        params = {
            'batch_size': batchsize,
            'shuffle': True,
            'num_workers': 2,
            'drop_last': True,
        }
        return DataLoader(train_dataset, **params), DataLoader(test_dataset, **params)

    def start(self):
        log.info("start training")
        self.__trainer.start_training()

def get_config():
    config = dict()

    torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = os.getenv('FIG_DEVICE')
    device = device if device is not None else torch_device
    config['device'] = device

    default_generated_pca_landmark_path = "/media/tuantran/rapid-data/dataset/LRW/attention-based-face-generation/generated_pca_landmark_8.pkl"
    generated_pca_landmark_path = os.getenv('FIG_GENERATED_PCA_LANDMARK_PATH')
    generated_pca_landmark_path = generated_pca_landmark_path if generated_pca_landmark_path is not None else default_generated_pca_landmark_path
    config['generated_pca_landmark_path'] = generated_pca_landmark_path

    default_original_landmark_path = "/media/tuantran/rapid-data/dataset/LRW/attention-based-face-generation/raw_landmark.pkl"
    original_landmark_path = os.getenv('FIG_ORIGINAL_LANDMARK_PATH')
    original_landmark_path = original_landmark_path if original_landmark_path is not None else default_original_landmark_path
    config['original_landmark_path'] = original_landmark_path

    default_face_root_path = "/media/tuantran/rapid-data/dataset/LRW/attention-based-face-generation/training_video"
    face_root_path = os.getenv('FIG_FACE_ROOT_PATH')
    face_root_path = face_root_path if face_root_path is not None else default_face_root_path
    config['face_root_path'] = face_root_path

    default_output_path = "./face_decoder_output"
    output_path = os.getenv('FIG_OUTPUT_PATH')
    output_path = output_path if output_path is not None else default_output_path
    config['output_path'] = output_path

    default_pretrained_model_folder = "./face_decoder_output/models/final"
    pretrained_model_folder = os.getenv('FIG_PRETRAINED_MODEL_FOLDER')
    pretrained_model_folder = pretrained_model_folder if pretrained_model_folder is not None else default_pretrained_model_folder
    pretrained_model_paths = {}
    if os.path.exists(pretrained_model_folder):
        lst = set(os.listdir(pretrained_model_folder))
        for name in ['generator', 'discriminator']:
            filename = name + '.pt'
            if filename not in lst:
                break
            pretrained_model_paths[name] = os.path.join(pretrained_model_folder, filename)
    if len(pretrained_model_paths) == 2:
        config['pretrained_model_paths'] = pretrained_model_paths

    default_epochs = 20
    epochs = os.getenv('FIG_EPOCHS')
    epochs = int(epochs) if epochs is not None else default_epochs
    config['epochs'] = epochs

    default_epoch_offset = 1
    epoch_offset = os.getenv('FIG_EPOCH_OFFSET')
    epoch_offset = int(epoch_offset) if epoch_offset is not None else default_epoch_offset
    config['epoch_offset'] = epoch_offset

    default_batchsize = 6
    batchsize = os.getenv('FIG_BATCHSIZE')
    batchsize = int(batchsize) if batchsize is not None else default_batchsize
    config['batchsize'] = batchsize

    return config

if __name__ == "__main__":
    config = get_config()
    log.info("Running with config: {}".format(config))
    trainer = FaceGeneratorTrainer(**config)
    trainer.start()
