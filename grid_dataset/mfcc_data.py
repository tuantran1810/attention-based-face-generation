import cv2, sys, os, copy, math, librosa
import numpy as np
from pickle import dump
import python_speech_features
from tqdm import tqdm

class MFCCData(object):
    def __init__(
        self,
        audiorootfolder = "./sample_audio",
        outputpath = "./preprocessed/mfcc.pkl",
        audio_ext = "wav",
        sound_features = 12,
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

        self.__sound_features = sound_features
        self.__outputpath = outputpath

    def __sound_mfcc(self, audio_path, nframes = 75):
        audio, sr = librosa.load(audio_path, sr = 16000)
        audio = np.append(audio, np.zeros(160))
        mfcc = python_speech_features.mfcc(audio, 16000, winstep = 0.01)
        return mfcc

    def run(self):
        final = {}
        for identity, identity_map in tqdm(self.__paths.items()):
            if identity not in final:
                final[identity] = {}
            for code, path in tqdm(identity_map.items()):
                mfcc = self.__sound_mfcc(path, nframes = 75)
                if mfcc.shape[0] != 300:
                    print("invalid audio: {}".format(path))
                    continue
                final[identity][code] = mfcc
                # print("processed for : {} --- {}".format(identity, code))
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
