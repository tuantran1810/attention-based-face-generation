import sys, os, random, torch, pickle, cv2, cpbd
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
from skimage.metrics import structural_similarity as compare_ssim

class GRIDValidator():
    def __init__(self,
        model_root_path = "./model/grid/epoch_10",
        batchsize = 4,
        landmark_features = 7,
        landmark_pca_path = "./grid_dataset/preprocessed/landmark_pca.pkl",
        landmark_mean_path = "./grid_dataset/preprocessed/standard_landmark_mean.pkl",
        dataset_path = "./grid_dataset/preprocessed/dataset_split.pkl",
        generated_pca_landmark_path = "/media/tuantran/raid-data/dataset/GRID/attention-based-face-generation/generated_pca_landmark_6_50.pkl",
        standard_landmark_path = "/media/tuantran/raid-data/dataset/GRID/standard_landmark.pkl",
        face_root_path = "/media/tuantran/rapid-data/dataset/GRID/face_images_128",
        output_path = "./face_decoder_output",
        device = "cpu",
    ):
        self.__device = device
        self.__output_path = output_path
        self.__dataset_path = dataset_path

        self.__generator = Generator(device = device)
        self.__discriminator = Discriminator(device = device)
        self.__generator.load_state_dict(torch.load(model_root_path + "/generator.pt"))
        self.__discriminator.load_state_dict(torch.load(model_root_path + "/discriminator.pt"))

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

        landmark_pca_metadata = landmark_pca_metadata[landmark_features]
        self.__landmark_pca_mean = np.expand_dims(landmark_pca_metadata["mean"], 0)
        self.__landmark_pca_components = landmark_pca_metadata["components"]

        _, self.__test_dataloader = self.__create_dataloader(generated_pca_landmark_path, standard_landmark_path, face_root_path, dataset_path, batchsize = batchsize)
        self.__ones_vector = Variable(torch.ones(batchsize), requires_grad = False).to(device)
        self.__zeros_vector = Variable(torch.zeros(batchsize), requires_grad = False).to(device)
        self.__landmark_loss_mask = torch.cat([torch.ones(96), torch.ones(40)*100]).view(1, 1, 136).to(device)

    def __data_processing(self, item):
        def process_pca_landmark(pca_landmarks, index):
            if index < 6:
                raise Exception("invalid index")
            index -= 6
            pca_landmark = pca_landmarks[index]
            landmark = np.matmul(pca_landmark, self.__landmark_pca_components) + self.__landmark_pca_mean
            landmark = landmark + self.__landmark_mean - 0.5
            landmark = torch.tensor(landmark).float()
            return landmark

        def process_inspired_landmark(standard_landmarks, index):
            if index < 6:
                raise Exception("invalid index")
            index -= 6
            landmark = standard_landmarks[index].flatten()
            landmark = landmark - 0.5
            landmark = torch.tensor(landmark).float()
            return landmark

        r = random.randint(6, 49)
        r_sequence, r_inspire = r, r
        window = 16
        landmarks = process_pca_landmark(item['generated_pca_landmark'], r_sequence)
        inspired_landmark = process_inspired_landmark(item['standard_landmark'], r_inspire)

        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
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

        inspired_image = face_images[r_inspire]
        face_images = face_images[r_sequence+1:r_sequence+1+window].permute(1,0,2,3)

        return ((item['identity'], item['code'], r_sequence+1), (landmarks, face_images), (inspired_landmark, inspired_image))

    def __create_dataloader(
        self,
        generated_pca_landmark_path,
        standard_landmark_path,
        root_face_video_path,
        dataset_path,
        batchsize,
        video_file_ext = 'gzip',
    ):
        random.seed(0)
        generated_landmark_data = None
        with open(generated_pca_landmark_path, 'rb') as fd:
            generated_landmark_data = pickle.load(fd)

        standard_landmark_data = None
        with open(standard_landmark_path, 'rb') as fd:
            standard_landmark_data = pickle.load(fd)

        dataset_split = None
        with open(dataset_path, 'rb') as fd:
            dataset_split = pickle.load(fd)

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

        train_data = list()
        val_data = list()

        train_dataset = dataset_split['train']
        val_dataset = dataset_split['val']
        for identity, code in train_dataset:
            train_data.append({
                'identity': identity,
                'code': code,
                'generated_pca_landmark': generated_landmark_data[identity][code], 
                'standard_landmark': standard_landmark_data[identity][code], 
                'video_path': face_video_paths[identity][code],
            })

        for identity, code in val_dataset:
            val_data.append({
                'identity': identity,
                'code': code,
                'generated_pca_landmark': generated_landmark_data[identity][code], 
                'standard_landmark': standard_landmark_data[identity][code], 
                'video_path': face_video_paths[identity][code],
            })

        train_dataset = ArrayDataset(train_data, self.__data_processing)
        test_dataset = ArrayDataset(val_data, self.__data_processing)
        params = {
            'batch_size': batchsize,
            'shuffle': True,
            'num_workers': 6,
            'drop_last': True,
        }
        return DataLoader(train_dataset, **params), DataLoader(test_dataset, **params)

    def __metric_log(self, metrics):
        lst = []
        for k, v in metrics.items():
            lst.append("{}: {:.4E}".format(k, v))
        body = ", ".join(lst)
        log.info(f"[metrics]: {body}")

    def start(self):
        log.info("start evaluating...")
        with torch.no_grad():
            cnt = 0.0
            pixel_loss_arr = []
            loss_generator_arr = []
            loss_generator_landmark_arr = []
            final_generator_loss_arr = []
            psnr_arr = []
            ssim_arr = []
            cpbd_arr = []
            i = 0
            for item in tqdm(self.__test_dataloader, "evaluating"):
                cnt += 1.0
                (identity, code, data_start_index), (landmarks, face_images), (inspired_landmark, inspired_image) = item
                landmarks = landmarks.to(self.__device)
                face_images = face_images.to(self.__device)
                inspired_landmark = inspired_landmark.to(self.__device)
                inspired_image = inspired_image.to(self.__device)

                attention_map, color_images, fake_images = self.__generator(inspired_image, inspired_landmark, landmarks)
                judgement, judgement_landmarks = self.__discriminator(fake_images, inspired_landmark)
                loss_generator = F.binary_cross_entropy(judgement, self.__ones_vector)
                loss_generator_landmark = F.mse_loss(
                    judgement_landmarks * self.__landmark_loss_mask,
                    landmarks * self.__landmark_loss_mask,
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

                face_images_np = (face_images.cpu().numpy() + 1.0)*127.5
                fake_images_np = (fake_images.cpu().numpy() + 1.0)*127.5

                def to_gray_arr(imgs):
                    imgs = imgs.transpose(0,2,3,4,1)
                    imgs = imgs.reshape((-1,128,128,3))
                    out = []
                    for img in imgs:
                        out.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                    out = np.stack(out)
                    return out

                # tmp = face_images_np.squeeze(0).squeeze(0)
                face_images_np = to_gray_arr(face_images_np)
                fake_images_np = to_gray_arr(fake_images_np)

                total_psnr = 0
                total_ssim = 0
                total_cpbd = 0
                for j in range(len(face_images_np)):
                    img1 = face_images_np[j]
                    img2 = fake_images_np[j]
                    psnr = cv2.PSNR(img1, img2)
                    ssim = compare_ssim(img1, img2)
                    # img_cpbd = cpbd.compute(img2)
                    total_psnr += psnr
                    total_ssim += ssim
                    # total_cpbd += img_cpbd
                psnr_arr.append(total_psnr/len(face_images_np))
                ssim_arr.append(total_ssim/len(face_images_np))
                cpbd_arr.append(total_cpbd/len(face_images_np))
                # self.__save_evaluation_data(
                #     i,
                #     identity,
                #     code,
                #     data_start_index,
                #     face_images,
                #     attention_map,
                #     color_images,
                #     fake_images
                # )
                i += 1

            metrics = {
                "pixel_loss": sum(pixel_loss_arr)/cnt,
                'loss_generator': sum(loss_generator_arr)/cnt,
                'loss_generator_landmark': sum(loss_generator_landmark_arr)/cnt,
                'final_generator_loss': sum(final_generator_loss_arr)/cnt,
                'psnr': sum(psnr_arr)/cnt,
                'ssim': sum(ssim_arr)/cnt,
                'cpbd': sum(cpbd_arr)/cnt,
            }
            self.__metric_log(metrics)

#orig: [metrics]: pixel_loss: 4.0330E-02, loss_generator: 1.2926E+00, loss_generator_landmark: 8.3518E-02, final_generator_loss: 1.7794E+00

#1: [metrics]: pixel_loss: 4.3949E-02, loss_generator: 6.2062E-01, loss_generator_landmark: 1.1743E-01, final_generator_loss: 1.1775E+00, psnr: 2.9999E+01, ssim: 7.0984E-01
#2: [metrics]: pixel_loss: 4.3507E-02, loss_generator: 1.0383E+00, loss_generator_landmark: 1.0441E-01, final_generator_loss: 1.5778E+00
#3: [metrics]: pixel_loss: 4.0582E-02, loss_generator: 7.7377E-01, loss_generator_landmark: 8.2418E-02, final_generator_loss: 1.2620E+00
#4: [metrics]: pixel_loss: 4.0920E-02, loss_generator: 1.0943E+00, loss_generator_landmark: 8.9524E-02, final_generator_loss: 1.5931E+00, psnr: 2.9859E+01, ssim: 7.1669E-01
#5: [metrics]: pixel_loss: 4.1881E-02, loss_generator: 6.9828E-01, loss_generator_landmark: 1.0984E-01, final_generator_loss: 1.2269E+00
#---->6: [metrics]: pixel_loss: 4.0020E-02, loss_generator: 1.0665E+00, loss_generator_landmark: 9.5382E-02, final_generator_loss: 1.5621E+00, psnr: 2.9860E+01, ssim: 7.2375E-01
#7: [metrics]: pixel_loss: 4.0743E-02, loss_generator: 1.2728E+00, loss_generator_landmark: 8.7627E-02, final_generator_loss: 1.7679E+00, psnr: 2.9706E+01, ssim: 7.1520E-01
#8: [metrics]: pixel_loss: 4.1367E-02, loss_generator: 1.0169E+00, loss_generator_landmark: 8.1971E-02, final_generator_loss: 1.5125E+00, psnr: 2.9711E+01, ssim: 7.1966E-01
#9: [metrics]: pixel_loss: 4.2412E-02, loss_generator: 1.1306E+00, loss_generator_landmark: 1.3608E-01, final_generator_loss: 1.6908E+00
#10: [metrics]: pixel_loss: 4.0252E-02, loss_generator: 1.1114E+00, loss_generator_landmark: 7.9859E-02, final_generator_loss: 1.5938E+00, psnr: 2.9827E+01, ssim: 7.2349E-01
#12: [metrics]: pixel_loss: 4.0221E-02, loss_generator: 9.4460E-01, loss_generator_landmark: 9.0851E-02, final_generator_loss: 1.4377E+00
#15: [metrics]: pixel_loss: 4.2582E-02, loss_generator: 1.3044E+00, loss_generator_landmark: 9.3017E-02, final_generator_loss: 1.8232E+00
#16: [metrics]: pixel_loss: 4.2939E-02, loss_generator: 1.7834E+00, loss_generator_landmark: 8.0020E-02, final_generator_loss: 2.2928E+00, psnr: 2.9490E+01, ssim: 7.1223E-01

#25: [metrics]: pixel_loss: 7.0229E-02, loss_generator: 6.9314E-01, loss_generator_landmark: 2.6726E-02, final_generator_loss: 1.4222E+00, psnr: 2.3956E+01, ssim: 5.3574E-01, cpbd: 5.5218E-02

if __name__ == "__main__":
    trainer = GRIDValidator(
        model_root_path = "./model/grid/epoch_7",
        device="cuda:0", 
        batchsize=100,
    )
    trainer.start()
