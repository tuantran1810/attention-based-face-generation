import sys, os, torch, pickle, cv2, cpbd
sys.path.append(os.path.dirname(__file__))
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
from networks.generator import Generator
from networks.discriminator import Discriminator
import numpy as np
from utils.dataset import ArrayDataset
from tqdm import tqdm
from pathlib import Path
from loguru import logger as log
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim

class LRWValidator():
    def __init__(self,
        model_root_path = "./model/lrw/epoch_20",
        batchsize = 4,
        dataset = "val",
        landmark_pca_path = "./lrw_dataset/preprocessed/landmark_pca_8.pkl",
        landmark_mean_path = "./lrw_dataset/preprocessed/standard_landmark_mean.pkl",
        generated_pca_landmark_path = "/media/tuantran/rapid-data/dataset/LRW/attention-based-face-generation/generated_pca_landmark_8.pkl",
        original_landmark_path = "/media/tuantran/rapid-data/dataset/LRW/attention-based-face-generation/raw_landmark.pkl",
        face_root_path = "/media/tuantran/rapid-data/dataset/LRW/attention-based-face-generation/training_video",
        output_path = "./lrw_validator_output",
        device = "cpu",
    ):
        self.__device = device
        self.__output_path = output_path

        self.__generator = Generator(device = device)
        self.__generator.load_state_dict(torch.load(model_root_path + '/generator.pt'))
        self.__discriminator = Discriminator(device = device)
        self.__discriminator.load_state_dict(torch.load(model_root_path + '/discriminator.pt'))

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

        self.__dataloader = self.__create_dataloader(dataset, generated_pca_landmark_path, original_landmark_path, face_root_path, batchsize = batchsize)
        self.__ones_vector = Variable(torch.ones(batchsize), requires_grad = False).to(device)
        self.__zeros_vector = Variable(torch.zeros(batchsize), requires_grad = False).to(device)
        self.__landmark_loss_mask = torch.cat([torch.ones(96), torch.ones(40)*100]).view(1, 1, 136).to(device)

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
        dataset,
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

        data = []
        for u, umap in generated_landmark_data.items():
            dmap = umap[dataset]
            for c, lm in dmap.items():
                olm = original_landmarks[u][dataset][c]
                data.append({
                    'udc': (u, dataset, c),
                    'vpath': os.path.join(root_face_video_path, u, dataset, c+'.mp4'),
                    'glm': lm, 
                    'olm': olm[4:27], 
                    'ilm': olm[3], 
                })

        data = ArrayDataset(data, self.__data_processing)
        params = {
            'batch_size': batchsize,
            'shuffle': True,
            'num_workers': 2,
            'drop_last': False,
        }
        return DataLoader(data, **params)


    def __save_evaluation_data(self,
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
        data_path = os.path.join(folder_path, "data_{}.pkl".format(sample))
        with open(data_path, 'wb') as fd:
            pickle.dump(data, fd)


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
            for item in tqdm(self.__dataloader, "evaluating"):
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
                    img_cpbd = cpbd.compute(img2)
                    total_psnr += psnr
                    total_ssim += ssim
                    total_cpbd += img_cpbd
                psnr_arr.append(total_psnr/len(face_images_np))
                ssim_arr.append(total_ssim/len(face_images_np))
                cpbd_arr.append(total_cpbd/len(face_images_np))
                # self.__save_evaluation_data(
                #     i,
                #     u,
                #     d,
                #     c,
                #     face_images,
                #     attention_map,
                #     color_images,
                #     fake_images
                # )

                i += 1
                if i == 25: break

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

#4: [metrics]: pixel_loss: 6.9703E-02, loss_generator: 6.9312E-01, loss_generator_landmark: 5.3283E-02, final_generator_loss: 1.4434E+00, psnr: 2.4007E+01, ssim: 5.3310E-01
#--->5: [metrics]: pixel_loss: 6.7928E-02, loss_generator: 6.9312E-01, loss_generator_landmark: 5.9484E-02, final_generator_loss: 1.4319E+00, psnr: 2.4167E+01, ssim: 5.4499E-01
#6: [metrics]: pixel_loss: 6.8855E-02, loss_generator: 6.9312E-01, loss_generator_landmark: 4.5132E-02, final_generator_loss: 1.4268E+00, psnr: 2.4025E+01, ssim: 5.3883E-01
#7: [metrics]: pixel_loss: 7.0275E-02, loss_generator: 6.9313E-01, loss_generator_landmark: 4.4048E-02, final_generator_loss: 1.4399E+00, psnr: 2.3900E+01, ssim: 5.3377E-01
#10: [metrics]: pixel_loss: 6.9027E-02, loss_generator: 6.9314E-01, loss_generator_landmark: 4.3510E-02, final_generator_loss: 1.4269E+00, psnr: 2.4122E+01, ssim: 5.4341E-01
#15: [metrics]: pixel_loss: 7.1139E-02, loss_generator: 6.9314E-01, loss_generator_landmark: 5.7004E-02, final_generator_loss: 1.4615E+00
#20: [metrics]: pixel_loss: 6.9944E-02, loss_generator: 6.9314E-01, loss_generator_landmark: 2.8448E-02, final_generator_loss: 1.4210E+00, psnr: 2.3959E+01, ssim: 5.4285E-01
#22 (300): [metrics]: pixel_loss: 6.9060E-02, loss_generator: 6.9314E-01, loss_generator_landmark: 5.7019E-02, final_generator_loss: 1.4408E+00
#23: [metrics]: pixel_loss: 6.9411E-02, loss_generator: 6.9314E-01, loss_generator_landmark: 5.6007E-02, final_generator_loss: 1.4433E+00
#24: [metrics]: pixel_loss: 6.8852E-02, loss_generator: 6.9314E-01, loss_generator_landmark: 5.7367E-02, final_generator_loss: 1.4390E+00
#25 (300): [metrics]: pixel_loss: 6.8502E-02, loss_generator: 6.9314E-01, loss_generator_landmark: 5.2412E-02, final_generator_loss: 1.4306E+00
#26: [metrics]: pixel_loss: 6.7477E-02, loss_generator: 6.9314E-01, loss_generator_landmark: 5.5513E-02, final_generator_loss: 1.4234E+00
#27: [metrics]: pixel_loss: 6.7951E-02, loss_generator: 6.9314E-01, loss_generator_landmark: 5.5926E-02, final_generator_loss: 1.4286E+00
#28 (300): [metrics]: pixel_loss: 6.9027E-02, loss_generator: 6.9314E-01, loss_generator_landmark: 5.3596E-02, final_generator_loss: 1.4370E+00
#30: [metrics]: pixel_loss: 7.0097E-02, loss_generator: 6.9314E-01, loss_generator_landmark: 5.1090E-02, final_generator_loss: 1.4452E+00
#35: [metrics]: pixel_loss: 6.8914E-02, loss_generator: 6.9312E-01, loss_generator_landmark: 4.9182E-02, final_generator_loss: 1.4314E+00
#40: [metrics]: pixel_loss: 6.9927E-02, loss_generator: 6.9314E-01, loss_generator_landmark: 2.6945E-02, final_generator_loss: 1.4194E+00, psnr: 2.3974E+01, ssim: 5.4410E-01


#25 (full): [metrics]: pixel_loss: 6.8031E-02, loss_generator: 6.9314E-01, loss_generator_landmark: 2.7422E-02, final_generator_loss: 1.4009E+00

if __name__ == "__main__":
    trainer = LRWValidator(
        model_root_path = "./model/lrw",
        dataset="test",
        device="cuda:0", 
        batchsize=100,
    )
    trainer.start()
