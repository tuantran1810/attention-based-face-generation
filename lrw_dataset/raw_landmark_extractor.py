import pickle
import cv2, sys, os, math, traceback, copy
sys.path.append(os.path.dirname(__file__))
import numpy as np
from pickle import dump, load
from tqdm import tqdm
import matplotlib.pyplot as plt
from mobilenet.mobile_net import MobileNetInference224

class RawLandmarkProcessor(object):
    def __init__(self, model_path = './mobilenet/mobilenet_224_model_best_gdconv_external.pth.tar', device = "cpu"):
        self.__landmark_detector = MobileNetInference224(
            model_path = model_path,
            device = device,
        )
        self.__landmark_size = 224

    def __resize_batch(self, frames, size):
        tmp_frames = []
        for frame in frames:
            frame = cv2.resize(frame, size)
            frame = np.expand_dims(frame, 0)
            tmp_frames.append(frame)
        frames = np.concatenate(tmp_frames, axis = 0)
        return frames

    def get_landmark(self, frames):
        frames = self.__resize_batch(frames, (self.__landmark_size, self.__landmark_size)).transpose(0,3,1,2)
        return self.__landmark_detector.infer_numpy(frames)


class RawLandmarkData(object):
    def __init__(
        self,
        video_avgrect_pkl = '/media/tuantran/raid-data/dataset/LRW/attention-based-face-generation/face_avg_rect.pkl',
        raw_videos = '/media/tuantran/rapid-data/dataset/LRW/attention-based-face-generation/training_video',
        output_pkl = '/media/tuantran/raid-data/dataset/LRW/attention-based-face-generation/raw_landmark_mobinet.pkl',
        device = "cpu",
    ):
        self.__processor = RawLandmarkProcessor(device = device)
        self.__output_pkl = output_pkl
        self.__result = dict()
        if os.path.exists(output_pkl):
            with open(output_pkl, 'rb') as fd:
                self.__result = pickle.load(fd)
        self.__raw_video_path = raw_videos
        self.__avg_rect = None
        with open(video_avgrect_pkl, 'rb') as fd:
            self.__avg_rect = load(fd)

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

    def run(self):
        result = {}
        cnt = 1
        for utterance, utterance_map in tqdm(self.__avg_rect.items()):
            if utterance not in self.__result: self.__result[utterance] = {}
            upath = os.path.join(self.__raw_video_path, utterance)
            for train_val_test, code_map in utterance_map.items():
                if train_val_test not in self.__result[utterance]: self.__result[utterance][train_val_test] = {}
                dpath = os.path.join(upath, train_val_test)
                for code, _ in code_map.items():
                    video_path = os.path.join(dpath, code + ".mp4")
                    if code in self.__result[utterance][train_val_test] and self.__result[utterance][train_val_test][code] is not None:
                        continue
                    
                    try:
                        frames = self.__video_frames(video_path)
                        if frames is None:
                            print("invalid video: {}".format(video_path))
                            continue
                        # frames = frames[:,l:r,t:b,:]
                        landmarks = self.__processor.get_landmark(frames)

                        # _, axes1 = plt.subplots(4,7)
                        # _, axes2 = plt.subplots(4,7)
                        # for i in range(28):
                        #     r, c = i//7,i%7
                        #     lm = landmarks[i]
                        #     axe1 = axes1[r][c]
                        #     axe2 = axes2[r][c]
                        #     axe1.imshow(frames[i])
                        #     axe2.scatter(lm[:,0], lm[:,1])
                        # plt.show()
                        # plt.close()

                        self.__result[utterance][train_val_test][code] = landmarks.astype(np.float32)
                    except Exception:
                        traceback.print_exc(file=sys.stdout)

                cnt += 1
                if cnt % 100 == 0:
                    with open(self.__output_pkl, 'wb') as fd:
                        dump(self.__result, fd)

        print("done")
        with open(self.__output_pkl, 'wb') as fd:
            dump(self.__result, fd)


def main():
    d = RawLandmarkData(device="cuda:0")
    d.run()

if __name__ == "__main__":
    main()
