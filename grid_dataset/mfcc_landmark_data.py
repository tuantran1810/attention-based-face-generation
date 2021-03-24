import os
import numpy as np
from sklearn.decomposition import PCA
from compress_pickle import load
from pickle import dump

class DataPCACalculator(object):
    def __init__(
        self,
        preprocessed_data = "./preprocessed",
        ext = "gzip",
        output_file = "./preprocessed/mfcc_landmark.pkl"
    ):
        '''
        self.__paths: map[identity]pathsList
        '''

        self.__paths = []

        for path, _ , files in os.walk(preprocessed_data):
            for name in files:
                code, file_ext = name.split('.')
                if file_ext == ext:
                    self.__paths.append(os.path.join(path, name))

        self.__output_file = output_file

    def run(self):
        lst = []
        total = len(self.__paths)
        one_percent = int(total / 100)
        for i, file in enumerate(self.__paths):
            mp = load(file, compression = 'gzip', set_default_extension = False)
            landmarks = mp['landmarks']
            mfcc = mp['mfcc']
            tmp = {}
            tmp['landmarks'] = landmarks
            tmp['mfcc'] = mfcc
            lst.append(tmp)
            if (i+1) % one_percent == 0:
                print("{} percent".format((i+1)/one_percent))

        with open(self.__output_file, "wb") as fd:
            dump(lst, fd)


def main():
    d = DataPCACalculator(preprocessed_data = "/media/tuantran/rapid-data/dataset/GRID/face_images_128")
    d.run()

if __name__ == "__main__":
    main()
