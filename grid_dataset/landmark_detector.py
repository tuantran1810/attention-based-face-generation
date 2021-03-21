import cv2
import sys
import os
from os import path
from pathlib import Path
import numpy
import json
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from mobilenet.mobile_net import MobileNetInference224
import numpy as np
import scipy.io.wavfile
import librosa
from compress_pickle import dump

class FaceLandmarkDetector(object):
    def __init__(
        self,
        videorootfolder = "./sample_videos",
        audiorootfolder = "./sample_audio",
        outputfolder = "./preprocessed",
        video_ext = "mpg",
        audio_ext = "wav",
        face_expected_ratio = "1:1",
        horizontal_landmark_scale = 2.2,
        output_w = 128,
        sound_features = 12,
    ):
        '''
        self.__paths: map[identity]pathsList
        '''

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

        for path, _ , files in os.walk(audiorootfolder):
            identity = path.split('/')[-1]
            for name in files:
                code, file_ext = name.split('.')
                if file_ext == audio_ext:
                    if identity in self.__paths and code in self.__paths[identity]:
                        video_path = self.__paths[identity][code]
                        lst = [video_path, os.path.join(path, name)]
                        self.__paths[identity][code] = lst

        Path(outputfolder).mkdir(parents = True, exist_ok = True)
        for path, _ , files in os.walk(outputfolder):
            identity = path.split('/')[-1]
            for name in files:
                code, file_ext = name.split('.')
                if file_ext == 'gzip':
                    if identity in self.__paths and code in self.__paths[identity]:
                        print('{}/{}.gzip existed'.format(identity, code))
                        self.__paths[identity].pop(code, None)

        self.__outputpath = outputfolder
        fw, fh = face_expected_ratio.split(':')
        self.__faceratiowh = int(fw)/int(fh)
        self.__output_w = output_w
        self.__output_h = int(output_w/self.__faceratiowh)

        self.__face_dectector = MTCNN(select_largest=True,keep_all=False, device='cuda:0')
        self.__horizontal_landmark_scale = horizontal_landmark_scale
        self.__landmark_detector = MobileNetInference224(
            model_path = './mobilenet/mobilenet_224_model_best_gdconv_external.pth.tar',
            device = 'cuda:0',
        )
        self.__landmark_size = 224
        self.__sound_features = sound_features

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

    def __face_landmark(self, frames, size):
        cropped_faces = self.__crop_face(frames)
        if cropped_faces is None:
            return None, None
        output_frames = self.__resize_batch(cropped_faces, size).transpose(0,3,1,2)
        lm_frames = self.__resize_batch(cropped_faces, (self.__landmark_size, self.__landmark_size)).transpose(0,3,1,2)
        landmarks = self.__landmark_detector.infer_numpy(lm_frames)
        return output_frames.astype(np.uint8), landmarks

    def __sound_mfcc(self, audio_path, nframes = 75):
        sr, audio = scipy.io.wavfile.read(audio_path)
        audio = audio.astype(float)
        chunksize = len(audio) // nframes
        output_lst = []
        for i in range(nframes):
            chunk = audio[i*chunksize:(i+1)*chunksize]
            s = librosa.feature.mfcc(audio[:chunksize], sr = sr, n_mfcc = self.__sound_features, fmax = 16000)
            s = np.expand_dims(s, axis = 0)
            output_lst.append(s)
        return np.concatenate(output_lst, 0)

    def __produce_file(self, outputpath, identity, code, faces, landmarks, mfcc):
        mp = {}
        mp['faces'] = faces
        mp['landmarks'] = landmarks
        mp['mfcc'] = mfcc
        filename = '{}.gzip'.format(code)
        filepath = os.path.join(outputpath, identity, filename)
        print(filepath)
        with open(filepath, 'wb') as fd:
            dump(mp, fd, compression = 'gzip')

    def run(self):
        for identity, identity_map in self.__paths.items():
            identity_path = os.path.join(self.__outputpath, identity)
            Path(identity_path).mkdir(parents = True, exist_ok = True)
            for code, lst in identity_map.items():
                video_path, audio_path = lst
                frames = self.__video_frames(video_path)
                if frames is None:
                    print("invalid video: {}".format(video_path))
                    continue
                faces, landmarks = self.__face_landmark(frames, (self.__output_w, self.__output_h))
                if faces is None or landmarks is None or faces.shape[0] != 75 or landmarks.shape[0] != 75:
                    print("invalid faces or landmarks: {}".format(video_path))
                    continue
                nfaces = frames.shape[0]
                mfcc = self.__sound_mfcc(audio_path, nframes = nfaces)
                if mfcc.shape[0] != 75:
                    print("invalid audio: {}".format(audio_path))
                    continue
                self.__produce_file(self.__outputpath, identity, code, faces, landmarks, mfcc)

def main():
    d = FaceLandmarkDetector(
        videorootfolder = "/media/tuantran/raid-data/dataset/GRID/video",
        audiorootfolder = "/media/tuantran/raid-data/dataset/GRID/audio_50",
        outputfolder = "/media/tuantran/rapid-data/dataset/GRID/face_images_128",
    )
    d.run()

if __name__ == "__main__":
    main()
