import cv2, sys, os, copy, math
import matplotlib.pyplot as plt
import numpy as np
from pickle import dump, load
from tqdm import tqdm
from scipy.spatial import procrustes


class LandmarkTransformation(object):
    def __init__(
        self, mean_face,
    ):
        self.__mean_shape = self.align_eye_points(np.expand_dims(mean_face, 0))[0]

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

    def align_eye_points(self, landmark_sequence):
        aligned_sequence = copy.deepcopy(landmark_sequence)
        eyecorner_dst = [ (np.float(0.3), np.float(1/3)), (np.float(0.7), np.float(1/3)) ]
        for i, landmark in enumerate(aligned_sequence):
            eyecorner_src  = [ (landmark[36, 0], landmark[36, 1]), (landmark[45, 0], landmark[45, 1]) ]
            transform, _ = self.__similarity_transform(eyecorner_src, eyecorner_dst)
            aligned_sequence[i] = self.__transform_landmark(landmark, transform)
        return aligned_sequence

    def transfer_expression(self, landmark_sequence):
        first_landmark = landmark_sequence[0,:,:]
        transform, _ = cv2.estimateAffine2D(first_landmark, self.__mean_shape, True)

        sx = np.sign(transform[0,0])*np.sqrt(transform[0,0]**2 + transform[0,1]**2)
        sy = np.sign(transform[1,0])*np.sqrt(transform[1,0]**2 + transform[1,1]**2)

        zero_vector = np.zeros((1, 68, 2))
        diff = np.cumsum(np.insert(np.diff(landmark_sequence, n=1, axis=0), 0, zero_vector, axis=0), axis=0)
        mean_shape_seq = np.tile(np.reshape(self.__mean_shape, (1, 68, 2)), [landmark_sequence.shape[0], 1, 1])

        diff[:, :, 0] = abs(sx)*diff[:, :, 0]
        diff[:, :, 1] = abs(sy)*diff[:, :, 1]

        transfer_expression_seq = diff + mean_shape_seq
        return np.float32(transfer_expression_seq)

    def get_mean_shape(self):
        return self.__mean_shape

    # def align_eyes_nose(self, landmark):


    def transfer_single_landmark(self, landmark):
        mtx1, mtx2, disparity = procrustes(self.__mean_shape, landmark)
        return mtx2

class LandmarkStandardize(object):
    def __init__(
        self,
        inputlandmark = "./preprocessed/raw_landmark.pkl",
        outputpath = "./preprocessed/standard_landmark.pkl",
    ):
        '''
        self.__paths: map[identity]pathsList
        '''
        self.__outputpath = outputpath
        self.__input_map = {}
        with open(inputlandmark, 'rb') as fd:
            self.__input_map = load(fd)
        del self.__input_map['mean']

        array = []
        for identity, identity_map in self.__input_map.items():
            for code, video_path in identity_map.items():
                landmark = self.__input_map[identity][code]
                landmark = np.mean(landmark, axis = 0)
                array.append(landmark)
        array = np.stack(array)
        mean_face = np.mean(array, axis = 0)
        self.__landmark_transformation = LandmarkTransformation(mean_face)

    def run(self):
        for identity, identity_map in tqdm(self.__input_map.items()):
            for code, video_path in tqdm(identity_map.items()):
                landmarks = self.__input_map[identity][code]
                landmarks = self.__landmark_transformation.align_eye_points(landmarks)
                output_landmarks = self.__landmark_transformation.transfer_expression(landmarks)
                self.__input_map[identity][code] = output_landmarks
        with open(self.__outputpath, 'wb') as fd:
            dump(self.__input_map, fd)

def main():
    d = LandmarkStandardize(
        inputlandmark = "/media/tuantran/raid-data/dataset/GRID/raw_landmark.pkl",
        outputpath = "/media/tuantran/raid-data/dataset/GRID/standard_landmark.pkl",
    )
    d.run()

if __name__ == "__main__":
    main()
