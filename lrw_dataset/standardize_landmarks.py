import pickle
import cv2, sys, os, dlib, math, traceback, copy
sys.path.append(os.path.dirname(__file__))
import numpy as np
from pickle import dump, load
from tqdm import tqdm
import matplotlib.pyplot as plt

class RawFaceDataProcessor(object):
    def __init__(
        self,
        standard_landmark_mean = './preprocessed/standard_landmark_mean.pkl',
    ):
        self.__standard_landmark_mean = None
        with open(standard_landmark_mean, 'rb') as fd:
            self.__standard_landmark_mean = load(fd)

        top_mouth = self.__standard_landmark_mean[51,1]
        bot_mouth = self.__standard_landmark_mean[57,1]
        top_lip = self.__standard_landmark_mean[62,1]
        bot_lip = self.__standard_landmark_mean[66,1]
        self.__mean_mouth_open_ratio = abs((bot_lip - top_lip)/(bot_mouth - top_mouth))

    def __similarity_transform(self, in_points, out_points):
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

    def __transform_landmark(self, landmark, transform):
        transformed = np.reshape(np.array(landmark), (68, 1, 2))
        transformed = cv2.transform(transformed, transform)
        transformed = np.float32(np.reshape(transformed, (68, 2)))
        return transformed

    def resize_batch(self, frames, size):
        tmp_frames = []
        for frame in frames:
            frame = cv2.resize(frame, size)
            frame = np.expand_dims(frame, 0)
            tmp_frames.append(frame)
        frames = np.concatenate(tmp_frames, axis = 0)
        return frames

    def align_eye_points(self, landmark_sequence):
        aligned_sequence = copy.deepcopy(landmark_sequence)
        eyecorner_dst = [ (np.float(0.3), np.float(1/3)), (np.float(0.7), np.float(1/3)) ]
        for i, landmark in enumerate(aligned_sequence):
            eyecorner_src  = [ (landmark[36, 0], landmark[36, 1]), (landmark[45, 0], landmark[45, 1]) ]
            transform, _ = self.__similarity_transform(eyecorner_src, eyecorner_dst)
            aligned_sequence[i] = self.__transform_landmark(landmark, transform)
        return aligned_sequence

    def align_nose_mouth(self, landmark_sequence):
        aligned_sequence = copy.deepcopy(landmark_sequence)
        dst = np.stack([
            self.__standard_landmark_mean[27],
            self.__standard_landmark_mean[30],
            self.__standard_landmark_mean[33],
            self.__standard_landmark_mean[8],
            self.__standard_landmark_mean[48],
            self.__standard_landmark_mean[54],
        ])
        for i, landmark in enumerate(aligned_sequence):
            src  = np.stack([
                landmark[27],
                landmark[30],
                landmark[33],
                landmark[8],
                landmark[48],
                landmark[54],
            ])
            transform, _ = self.__similarity_transform(src, dst)
            tmp = self.__transform_landmark(landmark, transform)
            landmark[27:36] = tmp[27:36]
            landmark[48:] = tmp[48:]
            aligned_sequence[i] = landmark
        return aligned_sequence

    def replace_landmark_boundary(self, landmarks):
        landmarks = copy.deepcopy(landmarks)
        arr = []
        for landmark in landmarks:
            landmark[:27] = self.__standard_landmark_mean[:27]
            arr.append(landmark)
        return np.stack(arr)

    def transfer_expression(self, landmark_sequence):
        # y_top_mouth = landmark_sequence[:,51,1]
        # y_bot_mouth = landmark_sequence[:,57,1]
        # mouth_openess = np.abs(y_bot_mouth - y_top_mouth)
        
        y_top_lip = landmark_sequence[:,62,1]
        y_bot_lip = landmark_sequence[:,66,1]
        lip_openess = np.abs(y_bot_lip - y_top_lip)
        # open_ratio = lip_openess/mouth_openess
        # tmp = np.abs(open_ratio - self.__mean_mouth_open_ratio)

        closest_landmark_to_mean = np.argmin(lip_openess)

        modeling_landmark = landmark_sequence[closest_landmark_to_mean,:,:]
        landmark_0 = landmark_sequence[0,:,:]
        transform, _ = cv2.estimateAffine2D(modeling_landmark, self.__standard_landmark_mean, True)

        landmark_sequence[0] = modeling_landmark
        landmark_sequence[closest_landmark_to_mean] = landmark_0
        sx = np.sign(transform[0,0])*np.sqrt(transform[0,0]**2 + transform[0,1]**2)
        sy = np.sign(transform[1,0])*np.sqrt(transform[1,0]**2 + transform[1,1]**2)

        zero_vector = np.zeros((1, 68, 2))
        lm_diff = np.diff(landmark_sequence, n=1, axis=0)
        lm_diff = np.insert(lm_diff, 0, zero_vector, axis=0)
        diff = np.cumsum(lm_diff, axis=0)
        mean_shape_seq = np.tile(np.reshape(self.__standard_landmark_mean, (1, 68, 2)), [landmark_sequence.shape[0], 1, 1])

        diff[:, :, 0] = abs(sx)*diff[:, :, 0]
        diff[:, :, 1] = abs(sy)*diff[:, :, 1]

        transfer_expression_seq = diff + mean_shape_seq

        modeling_landmark = transfer_expression_seq[0,:,:]
        landmark_0 = transfer_expression_seq[closest_landmark_to_mean,:,:]
        transfer_expression_seq[0] = landmark_0
        transfer_expression_seq[closest_landmark_to_mean] = modeling_landmark
        return np.float32(transfer_expression_seq)

    def process(self, landmarks):
        landmarks = self.align_eye_points(landmarks)
        mean_landmark = np.mean(landmarks, axis=0)
        landmark_trans = mean_landmark[27:48,:]
        mean_landmark_trans = self.__standard_landmark_mean[27:48,:]
        transformation, _ = self.__similarity_transform(landmark_trans, mean_landmark_trans)
        landmark_array = []
        for landmark in landmarks:
            landmark = self.__transform_landmark(landmark, transformation)
            landmark_array.append(landmark)
        landmarks = np.stack(landmark_array)
        landmarks = self.transfer_expression(landmarks)
        landmarks = self.align_nose_mouth(landmarks)
        landmarks = self.replace_landmark_boundary(landmarks)

        return landmarks.astype(np.float32)

class StandardizeLandmarks(object):
    def __init__(
        self,
        raw_landmark_pkl = '/media/tuantran/raid-data/dataset/LRW/attention-based-face-generation/raw_landmark_mobinet.pkl',
        standard_landmark_pkl = '/media/tuantran/raid-data/dataset/LRW/attention-based-face-generation/standard_landmark_mobinet_v2.pkl',
    ):
        self.__processor = RawFaceDataProcessor()

        self.__list = None
        with open(raw_landmark_pkl, 'rb') as fd:
            self.__list = load(fd)

        self.__standard_landmark_pkl = standard_landmark_pkl
        self.__result = None
        if os.path.exists(standard_landmark_pkl):
            with open(standard_landmark_pkl, 'rb') as fd:
                self.__result = load(fd)
        else:
            self.__result = dict()

    def __produce_file(self):
        with open(self.__standard_landmark_pkl, 'wb') as fd:
            pickle.dump(self.__result, fd)

    def run(self):
        cnt = 1
        for utterance, utterance_map in tqdm(self.__list.items()):
            if utterance not in self.__result: self.__result[utterance] = {}
            for train_val_test, code_map in utterance_map.items():
                if train_val_test not in self.__result[utterance]: self.__result[utterance][train_val_test] = {}
                for code, landmarks in code_map.items():
                    if code in self.__result[utterance][train_val_test] and self.__result[utterance][train_val_test][code] is not None:
                        continue
                    try:
                        standard_landmarks = self.__processor.process(landmarks)
                        self.__result[utterance][train_val_test][code] = standard_landmarks

                        # _,axes = plt.subplots(6,5)
                        # for i in range(29):
                        #     r,c = i//5, i%5
                        #     axe = axes[r][c]
                        #     slm = standard_landmarks[i]
                        #     lm = landmarks[i]
                        #     axe.scatter(lm[:,0] + 1.0, lm[:,1], c = 'b', s = 0.2)
                        #     axe.scatter(slm[:,0], slm[:,1], c = 'r', s = 0.2)
                        # plt.show()
                        # plt.close()
                    except Exception:
                        pass
                    self.__list[utterance][train_val_test][code] = None

            cnt += 1
            if cnt % 100 == 0:
                print(f"count = {cnt}, backup")
                self.__produce_file()
        print("done")
        self.__produce_file()


def main():
    d = StandardizeLandmarks()
    d.run()

if __name__ == "__main__":
    main()
