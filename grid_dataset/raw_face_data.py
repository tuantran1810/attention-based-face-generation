import cv2, sys, os, copy, math, librosa, scipy.io.wavfile
from os import path
from pathlib import Path
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from mobilenet.mobile_net import MobileNetInference224
import numpy as np
from pickle import dump
from compress_pickle import dump as cdump
from tqdm import tqdm

class RawFaceData(object):
    def __init__(
        self,
        videorootfolder = "./sample_videos",
        landmark_outputpath = "./preprocessed/raw_landmark.pkl",
        image_outputpath = "./preprocessed/images",
        video_ext = "mpg",
        face_expected_ratio = "1:1",
        horizontal_landmark_scale = 2.2,
        output_w = 128,
        device = 'cpu',
    ):
        '''
        self.__paths: map[identity]pathsList
        '''

        self.__device = device
        self.__paths = dict()

        for path, _ , files in os.walk(videorootfolder):
            identity = path.split('/')[-1]
            videomap = {}
            for name in files:
                code, file_ext = name.split('.')
                if file_ext == video_ext:
                    videomap[code] = os.path.join(path, name)
            if len(videomap) > 0:
                self.__paths[identity] = videomap

        Path(image_outputpath).mkdir(parents = True, exist_ok = True)
        for path, _ , files in os.walk(image_outputpath):
            identity = path.split('/')[-1]
            for name in files:
                code, file_ext = name.split('.')
                if file_ext == 'gzip':
                    if identity in self.__paths and code in self.__paths[identity]:
                        print('{}/{}.gzip existed'.format(identity, code))
                        self.__paths[identity].pop(code, None)

        self.__landmark_outputpath = landmark_outputpath
        self.__image_outputpath = image_outputpath
        fw, fh = face_expected_ratio.split(':')
        self.__faceratiowh = int(fw)/int(fh)
        self.__output_w = output_w
        self.__output_h = int(output_w/self.__faceratiowh)

        self.__face_dectector = MTCNN(select_largest=True,keep_all=False, device=self.__device)
        self.__horizontal_landmark_scale = horizontal_landmark_scale

    def __iterate_frames(self, videofile):
        vidcap = cv2.VideoCapture(videofile)
        while True:
            success, image = vidcap.read()
            if not success:
                return
            if image is None:
                print("image is None")
            yield image

    def __video_frames(self, videofile):
        frames = []
        for frame in self.__iterate_frames(videofile):
            frame = np.expand_dims(frame, axis = 0)
            frames.append(frame)
        frames = np.concatenate(frames, axis = 0)
        return frames

    def __resize_batch(self, frames, size):
        tmp_frames = []
        for frame in frames:
            frame = cv2.resize(frame, size)
            frame = np.expand_dims(frame, 0)
            tmp_frames.append(frame)
        frames = np.concatenate(tmp_frames, axis = 0)
        return frames

    def __crop_face(self, frames):
        _, _, landmarks = self.__face_dectector.detect(frames, landmarks = True)

        tmp_lmark = np.zeros((5, 2))
        cnt = 0.0
        for lm in landmarks:
            if lm is None:
                continue
            tmp_lmark += lm[0]
            cnt += 1.0
        if cnt < 1:
            return None

        tmp_lmark = tmp_lmark/cnt
        left_eye, right_eye, nose, left_mouth, right_mouth = tmp_lmark
        x_nose, y_nose = nose

        avg_left = (left_eye + left_mouth) / 2.0
        avg_right = (right_eye + right_mouth) / 2.0
        d_nose_left, _ = nose - avg_left
        d_nose_right, _ = avg_right - nose
        x1 = int(x_nose - self.__horizontal_landmark_scale*d_nose_left)
        x2 = int(x_nose + self.__horizontal_landmark_scale*d_nose_right)

        dx = x2 - x1
        dy = int(dx/self.__faceratiowh)

        avg_eye = (left_eye + right_eye) / 2.0
        avg_mouth = (left_mouth + right_mouth) / 2.0
        _, d_eyes_nose = nose - avg_eye
        _, d_nose_mouth = avg_mouth - nose
        d_nose_y1 = int(dy*(d_eyes_nose/(d_eyes_nose + d_nose_mouth)))
        d_nose_y2 = dy - d_nose_y1
        y1 = int(y_nose - d_nose_y1)
        y2 = int(y_nose + d_nose_y2)
        frames = frames[:,y1:y2,x1:x2,:]
        return frames

    def __produce_file(self, outputpath, faces, identity, code):
        image_mp = {}
        image_mp['identity'] = identity
        image_mp['code'] = code
        image_mp['faces'] = faces
        filename = '{}.gzip'.format(code)
        filepath = os.path.join(outputpath, identity, filename)
        with open(filepath, 'wb') as fd:
            cdump(image_mp, fd, compression = 'gzip')

    def run(self):
        for identity, identity_map in tqdm(self.__paths.items()):
            identity_path = os.path.join(self.__image_outputpath, identity)
            Path(identity_path).mkdir(parents = True, exist_ok = True)
            for code, video_path in tqdm(identity_map.items()):
                frames = self.__video_frames(video_path)
                if frames is None:
                    print("invalid video: {}".format(video_path))
                    continue
                faces = self.__crop_face(frames)
                if faces is None or faces.shape[0] != 75:
                    print("invalid faces or landmarks: {}".format(video_path))
                    continue
                faces = self.__resize_batch(faces, (self.__output_w, self.__output_h)).transpose(0,3,1,2)
                self.__produce_file(self.__image_outputpath, faces, identity, code)

def main():
    d = RawFaceData(
        videorootfolder = "/media/tuantran/raid-data/dataset/GRID/video",
        image_outputpath = "/media/tuantran/rapid-data/dataset/GRID/face_images_128",
        device = 'cuda:0',
    )
    d.run()

if __name__ == "__main__":
    main()
