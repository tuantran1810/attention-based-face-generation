import cv2, sys, os, copy, math, librosa, scipy.io.wavfile
from os import path
from pathlib import Path
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from mobilenet.mobile_net import MobileNetInference224
import numpy as np
from pickle import dump
from compress_pickle import load
from tqdm import tqdm

class RawLandmarkData(object):
    def __init__(
        self,
        imagerootfolder = "./preprocessed/images",
        landmark_outputpath = "./preprocessed/raw_landmark.pkl",
        image_ext = "gzip",
        device = 'cpu',
    ):
        '''
        self.__paths: map[identity]pathsList
        '''

        self.__device = device
        self.__paths = dict()

        for path, _ , files in os.walk(imagerootfolder):
            identity = path.split('/')[-1]
            imagemap = {}
            for name in files:
                code, file_ext = name.split('.')
                if file_ext == image_ext:
                    imagemap[code] = os.path.join(path, name)
            if len(imagemap) > 0:
                self.__paths[identity] = imagemap

        self.__landmark_outputpath = landmark_outputpath
        self.__landmark_detector = MobileNetInference224(
            model_path = './mobilenet/mobilenet_224_model_best_gdconv_external.pth.tar',
            device = self.__device,
        )
        self.__landmark_size = 224

    def __resize_batch(self, frames, size):
        tmp_frames = []
        for frame in frames:
            frame = cv2.resize(frame, size)
            frame = np.expand_dims(frame, 0)
            tmp_frames.append(frame)
        frames = np.concatenate(tmp_frames, axis = 0)
        return frames

    def __similarity_transform(self, in_points, out_points):
        s60 = math.sin(60*math.pi/180)
        c60 = math.cos(60*math.pi/180)

        in_points = np.copy(in_points).tolist()
        in_1_x, in_1_y = in_points[0]
        in_2_x, in_2_y = in_points[1]
        out_points = np.copy(out_points).tolist()
        out_1_x, out_1_y = out_points[0]
        out_2_x, out_2_y = out_points[1]

        xin = c60*(in_1_x - in_2_x) - s60*(in_1_y - in_2_y) + in_2_x
        yin = s60*(in_1_x - in_2_x) + c60*(in_1_y - in_2_y) + in_2_y
        in_points.append([xin, yin])

        xout = c60*(out_1_x - out_2_x) - s60*(out_1_y - out_2_y) + out_2_x
        yout = s60*(out_1_x - out_2_x) + c60*(out_1_y - out_2_y) + out_2_y
        out_points.append([xout, yout])

        return cv2.estimateAffine2D(np.array([in_points]), np.array([out_points]), False)

    def __transform_landmark(self, landmark, transform):
        transformed = np.reshape(np.array(landmark), (68, 1, 2))
        transformed = cv2.transform(transformed, transform)
        transformed = np.float32(np.reshape(transformed, (68, 2)))
        return transformed

    def __align_eye_points(self, landmark_sequence):
        aligned_sequence = copy.deepcopy(landmark_sequence)
        first_landmark = aligned_sequence[0,:,:]
        
        eyecorner_dst = [ (np.float(0.3), np.float(1/3)), (np.float(0.7), np.float(1/3)) ]
        eyecorner_src  = [ (first_landmark[36, 0], first_landmark[36, 1]), (first_landmark[45, 0], first_landmark[45, 1]) ]

        transform, _ = self.__similarity_transform(eyecorner_src, eyecorner_dst)

        for i, landmark in enumerate(aligned_sequence):
            aligned_sequence[i] = self.__transform_landmark(landmark, transform)

        return aligned_sequence

    def run(self):
        all_landmarks = {}
        sum_landmarks = np.zeros((68, 2))
        cnt = 0
        for identity, identity_map in tqdm(self.__paths.items()):
            identity_path = os.path.join(self.__landmark_outputpath, identity)
            all_landmarks[identity] = {}
            for code, image_path in tqdm(identity_map.items()):
                frames = None
                with open(image_path, 'rb') as fd:
                    data = load(fd, compression = 'gzip')
                    frames = data['faces']
                if frames is None:
                    print("invalid video: {}".format(image_path))
                    continue
                frames = self.__resize_batch(frames.transpose(0,2,3,1), (self.__landmark_size, self.__landmark_size)).transpose(0,3,1,2)
                orig_landmarks = self.__landmark_detector.infer_numpy(frames)
                landmarks = self.__align_eye_points(orig_landmarks)
                all_landmarks[identity][code] = landmarks
                slm = landmarks.sum(axis = 0)
                sum_landmarks += slm
                cnt += 75.0
        avg_landmark = sum_landmarks/cnt
        all_landmarks['mean'] = avg_landmark
        with open(self.__landmark_outputpath, 'wb') as fd:
            dump(all_landmarks, fd)

def main():
    d = RawLandmarkData(
        imagerootfolder = "/media/tuantran/rapid-data/dataset/GRID/face_images_128",
        device = 'cuda:0',
    )
    d.run()

if __name__ == "__main__":
    main()
