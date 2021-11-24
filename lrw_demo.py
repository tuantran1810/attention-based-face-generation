import sys, os
sys.path.append(os.path.dirname(__file__))

import cv2
import torch
import ffmpeg
import pickle
import librosa
import random
from copy import deepcopy
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from networks.generator import Generator
from networks.landmark_decoder import LandmarkDecoder
from utils.media import vidwrite
import scipy.io.wavfile as wav
from lrw_dataset.raw_video_preprocess import RawFaceDataProcessor
from lrw_dataset.mfcc_data import MFCCProcessor

class LRWDemo:
    def __init__(
        self,
        landmark_decoder_path,
        generator_path,
        images_path,
        audios_path,
        output_path,
        standard_landmark_mean_path = "./lrw_dataset/preprocessed/standard_landmark_mean.pkl",
        landmark_pca_path = "./lrw_dataset/preprocessed/landmark_pca_8.pkl",
        landmark_pca_features = 8,
        image_ext = "png",
        audio_ext = "wav",
        device = "cpu"
    ):
        self.__device = device
        self.__landmark_decoder = LandmarkDecoder(landmark_dims = landmark_pca_features, device = device)
        self.__landmark_decoder.load_state_dict(torch.load(landmark_decoder_path))
        self.__generator = Generator(device = device)
        self.__generator.load_state_dict(torch.load(generator_path))
        self.__raw_face_processor = RawFaceDataProcessor(
            standard_landmark_mean = standard_landmark_mean_path,
            landmark_predictor = "./lrw_dataset/shape_predictor_68_face_landmarks.dat",
        )
        self.__mfcc_processor = MFCCProcessor()

        self.__standard_landmark_mean = None
        with open(standard_landmark_mean_path, 'rb') as fd:
            self.__standard_landmark_mean = pickle.load(fd)

        pca_landmark_metadata = None
        with open(landmark_pca_path, 'rb') as fd:
            pca_landmark_metadata = pickle.load(fd)

        self.__landmark_pca_mean = pca_landmark_metadata["mean"]
        self.__landmark_pca_components = pca_landmark_metadata["components"]

        self.__data_paths = {}
        for path, _ , files in os.walk(images_path):
            for file in files:
                parts = file.split('.')
                if len(parts) != 2:
                    continue
                filename, ext = parts
                if ext != image_ext:
                    continue
                self.__data_paths[filename] = {
                    "image": os.path.join(path, file),
                }

        for path, _ , files in os.walk(audios_path):
            for file in files:
                parts = file.split('.')
                if len(parts) != 2:
                    continue
                filename, ext = parts
                if ext != audio_ext:
                    continue
                if filename not in self.__data_paths:
                    print(f"no image for: {filename}")
                    continue
                self.__data_paths[filename]["audio"] = os.path.join(path, file)

        self.__output_path = output_path
        self.__audio_padding_frames = 3
        self.__data = {}

    def preprocess(self):
        image_array = []
        landmark_array = []
        mfcc_array = []
        for key, value in self.__data_paths.items():
            if 'image' not in value or 'audio' not in value:
                print(f"{key} does not have enough image/audio")
                continue
            image = cv2.imread(value['image'])
            landmark, image = self.__raw_face_processor.frame_normalize(image)
            image = image.transpose(2,0,1)

            mfcc = self.__mfcc_processor.mfcc_from_path(value['audio'])
            mfcc = mfcc.transpose(1,0)[1:,:148]
            print(mfcc.shape)

            image_array.append(image)
            landmark_array.append(landmark)
            mfcc_array.append(mfcc)

        self.__data['images'] = np.stack(image_array)
        self.__data['landmarks'] = np.stack(landmark_array)
        self.__data['mfcc'] = np.stack(mfcc_array)

        return self

    def infer_landmark(self):
        landmarks = deepcopy(self.__data['landmarks'])
        mfccs = self.__data['mfcc']
        batchsize = landmarks.shape[0]
        landmarks -= self.__standard_landmark_mean
        landmarks = landmarks.reshape(batchsize, -1)
        landmarks -= self.__landmark_pca_mean
        landmarks = landmarks.dot(self.__landmark_pca_components.T)
        landmarks = torch.tensor(landmarks).to(self.__device)

        mfccs = torch.tensor(mfccs).to(self.__device)
        mfccs = mfccs.unsqueeze(1)
        generated_landmarks = None
        # self.__landmark_decoder.eval()
        with torch.no_grad():
            generated_landmarks = self.__landmark_decoder.forward(landmarks.float(), mfccs.float())
        generated_landmarks = generated_landmarks.to("cpu").detach().numpy()
        generated_landmarks = generated_landmarks.dot(self.__landmark_pca_components) + self.__landmark_pca_mean
        standard_landmark = self.__standard_landmark_mean.reshape(136)
        standard_landmark = np.expand_dims(np.expand_dims(standard_landmark, 0), 0)
        generated_landmarks += standard_landmark
        frames = generated_landmarks.shape[1]
        generated_landmarks = generated_landmarks.reshape(batchsize, frames, 68, -1)
        self.__data['generated_landmarks'] = generated_landmarks

        return self

    def to_image_sequence(self):
        inspired_landmark = deepcopy(self.__data['landmarks'])
        inspired_image = deepcopy(self.__data['images'])
        generated_landmarks = deepcopy(self.__data['generated_landmarks'])

        batchsize = generated_landmarks.shape[0]
        frames = generated_landmarks.shape[1]
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        inspired_landmark = inspired_landmark - 0.5
        inspired_landmark = inspired_landmark.reshape(batchsize, -1)
        inspired_landmark = torch.tensor(inspired_landmark).to(self.__device).float()

        image_array = []
        inspired_image = inspired_image.transpose(0, 2, 3, 1)
        for image in inspired_image:
            image = transform_ops(image)
            image_array.append(image)
        inspired_image = torch.stack(image_array)

        generated_landmarks = generated_landmarks - 0.5
        generated_landmarks = generated_landmarks.reshape(batchsize, frames, -1)
        generated_landmarks = torch.tensor(generated_landmarks).to(self.__device).float()

        attention_map, color_images, final_images = None, None, None
        # self.__generator.eval()
        with torch.no_grad():
            attention_map, color_images, final_images = self.__generator(inspired_image, inspired_landmark, generated_landmarks)

        attention_map = attention_map.detach().cpu().numpy()
        color_images = color_images.detach().cpu().numpy()
        final_images = final_images.detach().cpu().numpy()

        self.__data['attention_map'] = attention_map
        self.__data['color_images'] = np.uint8((color_images + 1.0) * 127.5)
        self.__data['final_images'] = np.uint8((final_images + 1.0) * 127.5)

        return self

    def to_image_file(self):
        attention_map = deepcopy(self.__data['attention_map'])
        color_images = deepcopy(self.__data['color_images'])
        final_images = deepcopy(self.__data['final_images'])

        plain_video_output_path = os.path.join(self.__output_path, "images")
        Path(plain_video_output_path).mkdir(parents=True, exist_ok=True)

        for i, k in enumerate(self.__data_paths.keys()):
            att = attention_map[i]
            color_image = color_images[i]
            final_image = final_images[i]

            att = (np.repeat(att.transpose(1,2,3,0), 3, axis = 3) * 255.0).astype(np.uint8)
            color_image = color_image.transpose(1,2,3,0)
            final_image = final_image.transpose(1,2,3,0)

            combined = np.concatenate(
                [
                    final_image,
                    color_image,
                    att,
                ],
                axis = 1,
            )

            video_frame_array = []
            for frame in combined:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_frame_array.append(frame)
            combined = np.stack(video_frame_array)

            tmp = [[],[],[]]
            for j in range(3):
                for t in range(23):
                    frame = combined[j*23+t]
                    tmp[j].append(frame)
                tmp[j] = np.concatenate(tmp[j], axis=1)
            img = np.concatenate(tmp, axis=0)

            figimg = plt.figure()
            plt.imshow(img)
            plt.axis("off")
            figimg.tight_layout()
            figimg.set_figheight(10)
            figimg.set_figwidth(20)
            figimg.savefig(os.path.join(plain_video_output_path, k+'.png'), dpi = 100)
        return self

    def to_video(self):
        attention_map = deepcopy(self.__data['attention_map'])
        color_images = deepcopy(self.__data['color_images'])
        final_images = deepcopy(self.__data['final_images'])

        plain_video_output_path = os.path.join(self.__output_path, "plain_video")
        Path(plain_video_output_path).mkdir(parents=True, exist_ok=True)

        combined_plain_videos = []
        for i, k in enumerate(self.__data_paths.keys()):
            att = attention_map[i]
            color_image = color_images[i]
            final_image = final_images[i]

            att = (np.repeat(att.transpose(1,2,3,0), 3, axis = 3) * 255.0).astype(np.uint8)
            color_image = color_image.transpose(1,2,3,0)
            final_image = final_image.transpose(1,2,3,0)

            combined = np.concatenate(
                [
                    final_image,
                    color_image,
                    att,
                ],
                axis = 2,
            )

            video_frame_array = []
            for frame in combined:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_frame_array.append(frame)
            combined = np.stack(video_frame_array)

            plain_video_path = os.path.join(plain_video_output_path, k+'.mp4')
            vidwrite(plain_video_path, combined, vcodec='libx264', fps=25)
            combined_plain_videos.append(combined)
        self.__data['combined_plain_videos'] = np.stack(combined_plain_videos)
        return self

    def to_audio(self):
        combined_plain_videos = self.__data['combined_plain_videos']
        audio_output_path = os.path.join(self.__output_path, "audio")
        Path(audio_output_path).mkdir(parents=True, exist_ok=True)
        batchsize, frames, _, _, _ = combined_plain_videos.shape
        audio_frames = frames + 2*self.__audio_padding_frames
        for i, k in enumerate(self.__data_paths.keys()):
            audio_src_file = self.__data_paths[k]['audio']
            audio, sr = librosa.load(audio_src_file, sr = 50000)
            audio_samples_per_frame = audio.shape[0]//audio_frames
            audio_start_point = audio_samples_per_frame * self.__audio_padding_frames
            audio_stop_point = audio.shape[0] - (audio_samples_per_frame * self.__audio_padding_frames)
            audio = audio[audio_start_point:audio_stop_point]
            wav.write(os.path.join(audio_output_path, k+'.wav'), sr, audio) 
        return self


    def to_final_video(self):
        plain_video_output_path = os.path.join(self.__output_path, "plain_video")
        audio_output_path = os.path.join(self.__output_path, "audio")
        final_output_path = os.path.join(self.__output_path, "final")
        Path(final_output_path).mkdir(parents=True, exist_ok=True)
        for i, k in enumerate(self.__data_paths.keys()):
            audio_path = os.path.join(audio_output_path, k+'.wav')
            plain_video_path = os.path.join(plain_video_output_path, k+'.mp4')
            v = ffmpeg.input(plain_video_path)
            a = ffmpeg.input(audio_path)
            final = ffmpeg.output(v['v'], a['a'], os.path.join(final_output_path, k+'.mp4'), loglevel="panic")
            final.run()
        return self


def main():
    landmark_decoder_path = "./model/lrw/landmark_decoder.pt"
    generator_path = "./model/lrw/generator.pt"
    images_path = "./demo_lrw/images/"
    audios_path = "./demo_lrw/audios/"
    output_path = "./demo_lrw/output/"

    demo = LRWDemo(
        landmark_decoder_path,
        generator_path,
        images_path,
        audios_path,
        output_path,
        device = "cuda:0"
    )

    demo\
    .preprocess()\
    .infer_landmark()\
    .to_image_sequence()\
    .to_video()\
    .to_audio()\
    .to_final_video()
    # .to_image_file()\


if __name__ == "__main__":
    main()
