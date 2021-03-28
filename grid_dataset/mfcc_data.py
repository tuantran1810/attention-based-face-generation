import cv2, sys, os, copy, math, librosa, scipy.io.wavfile
import numpy as np
from pickle import dump

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

    def run(self):
        final = {}
        for identity, identity_map in self.__paths.items():
            if identity not in final:
                final[identity] = {}
            for code, path in identity_map.items():
                mfcc = self.__sound_mfcc(path, nframes = 75)
                if mfcc.shape[0] != 75:
                    print("invalid audio: {}".format(path))
                    continue
                final[identity][code] = mfcc
                print("processed for : {} --- {}".format(identity, code))
            with open(self.__outputpath, 'wb') as fd:
                dump(final, fd)

def main():
    d = MFCCData(
        audiorootfolder = "/media/tuantran/raid-data/dataset/GRID/audio_50",
        outputpath = "./preprocessed/mfcc.pkl",
    )
    d.run()

if __name__ == "__main__":
    main()
