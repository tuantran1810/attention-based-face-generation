import cv2, sys, os, copy, math, librosa
import numpy as np
from pickle import dump
from tqdm import tqdm

class MelData(object):
    def __init__(
        self,
        audiorootfolder = "./sample_audio",
        outputpath = "./preprocessed/mel.pkl",
        audio_ext = "wav",
        w = 0.04,
        h = 0.04,
    ):
        self.__paths = dict()

        for path, _ , files in os.walk(audiorootfolder):
            identity = path.split('/')[-1]
            audiomap = {}
            for name in files:
                code, file_ext = name.split('.')
                if file_ext == audio_ext:
                    audiomap[code] = os.path.join(path, name)
            if len(audiomap) > 0:
                self.__paths[identity] = audiomap

        self.__w = w
        self.__h = h
        self.__outputpath = outputpath

    def __sound_mel(self, audio_path, w, h):
        audio, sr = librosa.load(audio_path, sr = 50000)
        audio = audio.astype(float)
        cnst = 1 + (int(sr * w) / 2)
        y_stft_abs = np.abs(
            librosa.stft(
                audio,
                win_length = int(sr * w),
                hop_length = int(sr * h),
                n_fft = int(sr * w)
            ),
        ) / cnst

        melspec = np.log(
            1e-16 + librosa.feature.melspectrogram(
                sr = sr, 
                S = y_stft_abs**2,
                n_mels = 64
            )
        )
        return np.transpose(melspec)

    def run(self):
        mel_map = {}
        for identity, identity_map in tqdm(self.__paths.items()):
            if identity not in mel_map:
                mel_map[identity] = {}
            for code, path in tqdm(identity_map.items()):
                mel = self.__sound_mel(path, w = self.__w, h = self.__h)
                if mel.shape[0] != 76:
                    print("invalid audio: {}".format(path))
                    continue
                mel_map[identity][code] = mel
        with open(self.__outputpath, 'wb') as fd:
            dump(mel_map, fd)

def main():
    d = MelData(
        audiorootfolder = "/media/tuantran/raid-data/dataset/GRID/audio_50",
        outputpath = "/media/tuantran/raid-data/dataset/GRID/audio_50/mel.pkl",
    )
    d.run()

if __name__ == "__main__":
    main()
