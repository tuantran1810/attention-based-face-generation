import os,librosa
import numpy as np
from pickle import dump, load
import python_speech_features
from tqdm import tqdm

class MFCCProcessor():
    def __init__(self, sampling_rate = 8000):
        self.__sampling_rate = sampling_rate

    def to_mfcc(self, audio):
        audio = np.append(audio, np.zeros(self.__sampling_rate))
        audio = np.append(np.zeros(960), audio)
        mfcc = python_speech_features.mfcc(audio, self.__sampling_rate, winstep = 0.01)
        return mfcc

    def mfcc_from_path(self, audio_path):
        audio, sr = librosa.load(audio_path, sr = self.__sampling_rate)
        return self.to_mfcc(audio)

class MFCCData(object):
    def __init__(
        self,
        audiorootfolder = "./preprocessed/plain_audio",
        outputpath = "./preprocessed/mfcc.pkl",
        audio_ext = "wav",
    ):
        self.__output = None
        self.__paths = dict()
        if os.path.exists(outputpath):
            with open(outputpath, 'rb') as fd:
                self.__output = load(outputpath)
        else:
            self.__output = dict()

        for path, _ , files in os.walk(audiorootfolder):
            if len(files) == 0:
                continue

            segments = path.split('/')
            utterance = segments[-2]
            train_val_test = segments[-1]

            if utterance not in self.__output:
                self.__output[utterance] = dict()
                self.__output[utterance][train_val_test] = dict()
                self.__paths[utterance] = dict()
                self.__paths[utterance][train_val_test] = dict()
            elif train_val_test not in self.__output[utterance]:
                self.__output[utterance][train_val_test] = dict()
                self.__paths[utterance][train_val_test] = dict()

            for name in files:
                code, file_ext = name.split('.')
                if file_ext == audio_ext:
                    if code not in self.__output[utterance][train_val_test] or self.__output[utterance][train_val_test][code] is None:
                        self.__output[utterance][train_val_test][code] = None
                        self.__paths[utterance][train_val_test][code] = os.path.join(path, name)

        self.__outputpath = outputpath
        self.__processor = MFCCProcessor()

    def run(self):
        print("start processing")
        for utterance, utterance_map in tqdm(self.__paths.items()):
            for train_val_test, code_map in utterance_map.items():
                for code in code_map:
                    if code_map[code] is None:
                        continue
                    mfcc = self.__processor.mfcc_from_path(code_map[code])
                    self.__output[utterance][train_val_test][code] = mfcc

        with open(self.__outputpath, 'wb') as fd:
            dump(self.__output, fd)

def main():
    d = MFCCData(
        audiorootfolder = "/media/tuantran/raid-data/dataset/LRW/attention-based-face-generation/plain_audio",
        outputpath = "/media/tuantran/raid-data/dataset/LRW/attention-based-face-generation//mfcc.pkl",
    )
    d.run()

if __name__ == "__main__":
    main()
