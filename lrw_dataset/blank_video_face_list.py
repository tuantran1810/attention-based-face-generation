import sys, os
sys.path.append(os.path.dirname(__file__))
import pickle

class BlankVideoFaceList(object):
    def __init__(
        self,
        videorootfolder = "./sample_videos",
        video_ext = "mp4",
        output_folder = "./preprocessed/",
        n_parts = 2,
    ):
        self.__paths = dict()

        for path, _ , files in os.walk(videorootfolder):
            if len(files) == 0:
                continue

            segments = path.split('/')
            utterance = segments[-2]
            train_val_test = segments[-1]

            if utterance not in self.__paths:
                self.__paths[utterance] = dict()
                self.__paths[utterance][train_val_test] = dict()
            elif train_val_test not in self.__paths[utterance]:
                self.__paths[utterance][train_val_test] = dict()

            videomap = dict()
            for name in files:
                code, file_ext = name.split('.')
                if file_ext == video_ext:
                    videomap[code] = {}
                    videomap[code]['path'] = os.path.join(path, name)
                    videomap[code]['ltrb'] = None
                    videomap[code]['landmark'] = None
            self.__paths[utterance][train_val_test] = videomap

        self.__n_parts = n_parts
        self.__output_folder = output_folder

    def run(self):
        lst = []
        for i in range(self.__n_parts):
            lst.append(dict())
        keys = self.__paths.keys()
        for i, key in enumerate(keys):
            i = i%self.__n_parts
            d = lst[i]
            d[key] = self.__paths[key]
        for i, d in enumerate(lst):
            path = os.path.join(self.__output_folder, f"list_{i}.pkl")
            with open(path, 'wb') as fd:
                pickle.dump(d, fd)

if __name__ == "__main__":
    ops = BlankVideoFaceList(
        # videorootfolder = "/media/tuantran/raid-data/dataset/LRW",
        # output_folder = "/media/tuantran/raid-data/dataset/LRW/attention-based-face-generation",
        # n_parts = 6,
    )
    ops.run()
