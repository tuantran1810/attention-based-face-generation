import cv2, sys, os, math, copy
sys.path.append(os.path.dirname(__file__))
from pathlib import Path
import numpy as np
from pickle import load
from tqdm import tqdm
import matplotlib.pyplot as plt

def similarity_transform(in_points, out_points):
    s60 = math.sin(60*math.pi/180)
    c60 = math.cos(60*math.pi/180)

    in_points = np.copy(in_points).tolist()
    in_1_x, in_1_y = in_points[0]
    in_2_x, in_2_y = in_points[1]
    out_points = np.copy(out_points).tolist()
    out_1_x, out_1_y = out_points[0]
    out_2_x, out_2_y = out_points[1]

    xin = c60*(in_1_x - in_2_x) - s60*(in_1_y - in_2_y) + in_2_x
    yin = s60*(in_1_x - in_2_x) + c60*(in_1_y - in_2_y) + in_2_y
    in_points.append([xin, yin])

    xout = c60*(out_1_x - out_2_x) - s60*(out_1_y - out_2_y) + out_2_x
    yout = s60*(out_1_x - out_2_x) + c60*(out_1_y - out_2_y) + out_2_y
    out_points.append([xout, yout])

    return cv2.estimateAffine2D(np.array([in_points]), np.array([out_points]), False)

def transform_landmark(landmark, transform):
    transformed = np.reshape(np.array(landmark), (68, 1, 2))
    transformed = cv2.transform(transformed, transform)
    transformed = np.float32(np.reshape(transformed, (68, 2)))
    return transformed

def align_eye_points(landmark_sequence):
    aligned_sequence = copy.deepcopy(landmark_sequence)
    eyecorner_dst = [ (np.float(0.3), np.float(1/3)), (np.float(0.7), np.float(1/3)) ]
    for i, landmark in enumerate(aligned_sequence):
        eyecorner_src  = [ (landmark[36, 0], landmark[36, 1]), (landmark[45, 0], landmark[45, 1]) ]
        transform, _ = similarity_transform(eyecorner_src, eyecorner_dst)
        aligned_sequence[i] = transform_landmark(landmark, transform)
    return aligned_sequence

def in_range(num, nmax, nmin):
    return max(min(num, nmax), nmin)

def write_video(video, path, fps = 25):
    f, w, h, _ = video.shape
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
    for i in range(f):
        data = video[i]
        out.write(data)
    out.release()

class VideoStandardize(object):
    def __init__(
        self,
        video_root_path = '/media/tuantran/raid-data/dataset/LRW/raw',
        video_output_path = '/media/tuantran/rapid-data/dataset/LRW/attention-based-face-generation/training_video',
        raw_landmark_pkl = '/media/tuantran/rapid-data/dataset/LRW/attention-based-face-generation/raw_landmark.pkl',
        standard_landmark_pkl = '/media/tuantran/rapid-data/dataset/LRW/attention-based-face-generation/standard_landmark.pkl',
        face_rect_path_pkl = '/media/tuantran/rapid-data/dataset/LRW/attention-based-face-generation/face_avg_rect.pkl',
    ):
        self.__video_root_path = video_root_path
        self.__video_output_path = video_output_path

        self.__raw_landmark = None
        self.__standard_landmark = None
        self.__avg_rect = None
        with open(raw_landmark_pkl, 'rb') as fd:
            self.__raw_landmark = load(fd)
        with open(standard_landmark_pkl, 'rb') as fd:
            self.__standard_landmark = load(fd)
        with open(face_rect_path_pkl, 'rb') as fd:
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

    def run(self):
        for utterance, utterance_map in tqdm(self.__standard_landmark.items()):
            upath = os.path.join(self.__video_output_path, utterance)
            Path(upath).mkdir(parents = True, exist_ok = True)
            for dataset, code_map in utterance_map.items():
                dpath = os.path.join(upath, dataset)
                Path(dpath).mkdir(parents = True, exist_ok = True)
                for code, standard_landmark in code_map.items():
                    try:
                        filename = os.path.join(dpath, code + ".mp4")
                        if os.path.isfile(filename):
                            continue
                        raw_landmark = self.__raw_landmark[utterance][dataset][code]
                        raw_landmark = align_eye_points(raw_landmark)
                        ltrb = self.__avg_rect[utterance][dataset][code]
                        l, t, r, b = ltrb
                        rect = np.array([[l,t], [l,b], [r, b], [r,t]])
                        rect = np.expand_dims(rect, 1)
                        video_path = os.path.join(self.__video_root_path, utterance, dataset, code + '.mp4')
                        tframes = []
                        # oframes = []
                        for i, frame in enumerate(self.__iterate_frames(video_path)):
                            slm = standard_landmark[i]
                            slm = np.stack([
                                slm[27],
                                slm[30],
                                slm[33],
                                slm[48],
                                slm[54],
                            ])
                            rlm = raw_landmark[i]
                            rlm = np.stack([
                                rlm[27],
                                rlm[30],
                                rlm[33],
                                rlm[48],
                                rlm[54],
                            ])

                            transformation, _ = cv2.estimateAffinePartial2D(slm, rlm, False)
                            tframe = cv2.warpAffine(frame, transformation, (256, 256))
                            trect = cv2.transform(rect, transformation)

                            # frame = frame[l:r, t:b, :]
                            # frame = cv2.resize(frame, (256, 256))

                            trect = np.squeeze(trect, axis = 1)
                            center = np.mean(trect, axis = 0).astype(np.int)
                            xleft = (trect[0][0] + trect[1][0])//2 + 1
                            xright = (trect[2][0] + trect[3][0])//2 + 1
                            w = xright - xleft
                            ytop = (trect[0][1] + trect[3][1])//2 + 1
                            ybottom = (trect[1][1] + trect[2][1])//2 + 1
                            h = ybottom - ytop
                            tl, tr, tt, tb = in_range(center[0] - w//2, 256, 0), in_range(center[0] + w//2, 256, 0), in_range(center[1] - h//2, 256, 0), in_range(center[1] + h//2, 256, 0)
                            tframe = tframe[tt:tb, tl:tr, :]
                            tframe = cv2.resize(tframe, (128,128))

                            # oframes.append(frame)
                            tframes.append(tframe)

                        # oframes = np.stack(oframes)
                        tframes = np.stack(tframes)
                        # frames = np.concatenate([tframes, oframes], axis=2)

                        # _, axes = plt.subplots(4,7)
                        # for i in range(28):
                        #     r, c = i//7, i%7
                        #     axe = axes[r][c]
                        #     frame = tframes[i]
                        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        #     axe.imshow(frame)
                        # plt.show()
                        # plt.close()
                        write_video(tframes, filename)
                    except Exception:
                        print(f"exception throw on file {filename}")
                        pass

        print("done")

def main():
    d = VideoStandardize()
    d.run()

if __name__ == "__main__":
    main()
