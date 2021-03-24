import os
import numpy as np
from sklearn.preprocessing import normalize
from compress_pickle import load
from pickle import dump

class DataMFCCNormCalculator(object):
    def __init__(
        self,
        preprocessed_data = "./preprocessed",
        ext = "gzip",
        output_file = "./preprocessed/grid_mfcc_norm.pkl"
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
            mfcc = mp['mfcc']
            frames = mfcc.shape[0]
            mfcc = mfcc.reshape(frames, -1)
            lst.append(mfcc)
            if (i+1) % one_percent == 0:
                print("{} percent".format((i+1)/one_percent))

        final = np.concatenate(lst, axis = 0)
        print(final.shape)
        max_num = final.max()
        min_num = final.min()

        print(max_num, min_num)
        result = {}
        result["max"] = max_num
        result["min"] = min_num
        with open(self.__output_file, "wb") as fd:
            dump(result, fd)


def main():
    d = DataMFCCNormCalculator(preprocessed_data = "/media/tuantran/rapid-data/dataset/GRID/face_images_128")
    # d = DataMFCCNormCalculator(preprocessed_data = "./preprocessed")
    d.run()

if __name__ == "__main__":
    main()
