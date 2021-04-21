import cv2, sys, os, copy, math, librosa
import numpy as np
from pickle import dump
import python_speech_features
from tqdm import tqdm

class MFCCProcessor():
    def __init__(self, sampling_rate = 16000):
        self.__sampling_rate = sampling_rate

    def to_mfcc(self, audio):
        audio = np.append(audio, np.zeros(self.__sampling_rate//100))
        mfcc = python_speech_features.mfcc(audio, self.__sampling_rate, winstep = 0.01)
        return mfcc

    def mfcc_from_path(self, audio_path):
        audio, sr = librosa.load(audio_path, sr = self.__sampling_rate)
        return self.to_mfcc(audio)

class MFCCData(object):
    def __init__(
        self,
        audiorootfolder = "./sample_audio",
        outputpath = "./preprocessed/mfcc.pkl",
        audio_ext = "wav",
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

        self.__outputpath = outputpath
        self.__processor = MFCCProcessor()

    def run(self):
        final = {}
        for identity, identity_map in tqdm(self.__paths.items()):
            if identity not in final:
                final[identity] = {}
            for code, path in tqdm(identity_map.items()):
                mfcc = self.__processor.mfcc_from_path(path)
                if mfcc.shape[0] != 300:
                    print("invalid audio: {}".format(path))
                    continue
                final[identity][code] = mfcc
        with open(self.__outputpath, 'wb') as fd:
            dump(final, fd)

def main():
    d = MFCCData(
        audiorootfolder = "/media/tuantran/raid-data/dataset/GRID/audio_50",
        outputpath = "/media/tuantran/raid-data/dataset/GRID/audio_50/mfcc.pkl",
    )
    d.run()

if __name__ == "__main__":
    main()
