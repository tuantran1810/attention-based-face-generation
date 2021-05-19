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
        self.__frontal_face_detector = dlib.get_frontal_face_detector()
        self.__landmark_detector = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
        self.__standard_landmark_mean = None
        with open(standard_landmark_mean, 'rb') as fd:
            self.__standard_landmark_mean = load(fd)

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

    def __calculate_avg_rect(self, rect_array):
        tlx, tly, brx, bry = 0,0,0,0
        cnt = 0
        for rect in rect_array:
            if len(rect) == 0:
                continue
            rect = rect[0]
            tl = rect.tl_corner()
            br = rect.br_corner()
            tlx += tl.x
            tly += tl.y
            brx += br.x
            bry += br.y
            cnt += 1
        tlx, tly, brx, bry = int(tlx/cnt), int(tly/cnt), int(brx/cnt), int(bry/cnt)
        return dlib.rectangle(tlx, tly, brx, bry)

    def __detect_landmark(self, frame, rect, dtype="int"):
        shape = self.__landmark_detector(frame, rect)
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        tl = rect.tl_corner()
        br = rect.br_corner()
        center = np.array([(tl.x + br.x)/2, (tl.y + br.y)/2])
        wh = np.array([br.x - tl.x, br.y - tl.y])
        coords = (coords - center)/wh + 0.5
        face = frame[tl.x:br.x,tl.y:br.y]
        return coords, cv2.resize(face, (128,128))

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
            self.__standard_landmark_mean[48],
            self.__standard_landmark_mean[54],
        ])
        for i, landmark in enumerate(aligned_sequence):
            src  = np.stack([
                landmark[27],
                landmark[30],
                landmark[33],
                landmark[48],
                landmark[54],
            ])
            transform, _ = self.__similarity_transform(src, dst)
            tmp = self.__transform_landmark(landmark, transform)
            landmark[27:36] = tmp[27:36]
            landmark[48:] = tmp[48:]
            aligned_sequence[i] = landmark
        return aligned_sequence

    def transfer_expression(self, landmark_sequence):
        first_landmark = landmark_sequence[0,:,:]
        transform, _ = cv2.estimateAffine2D(first_landmark, self.__standard_landmark_mean, True)

        sx = np.sign(transform[0,0])*np.sqrt(transform[0,0]**2 + transform[0,1]**2)
        sy = np.sign(transform[1,0])*np.sqrt(transform[1,0]**2 + transform[1,1]**2)

        zero_vector = np.zeros((1, 68, 2))
        diff = np.cumsum(np.insert(np.diff(landmark_sequence, n=1, axis=0), 0, zero_vector, axis=0), axis=0)
        mean_shape_seq = np.tile(np.reshape(self.__standard_landmark_mean, (1, 68, 2)), [landmark_sequence.shape[0], 1, 1])

        diff[:, :, 0] = abs(sx)*diff[:, :, 0]
        diff[:, :, 1] = abs(sy)*diff[:, :, 1]

        transfer_expression_seq = diff + mean_shape_seq
        return np.float32(transfer_expression_seq)

    def crop_face(self, frames, avg_rect = None):
        if avg_rect is None:
            rect_array = []
            for frame in frames:
                rect = self.__frontal_face_detector(frame, 1)
                rect_array.append(rect)
            avg_rect = self.__calculate_avg_rect(rect_array)

        landmark_array = []
        face_array = []
        for frame in frames:
            landmark, face = self.__detect_landmark(frame, avg_rect)
            landmark_array.append(landmark)
            face_array.append(face)

        landmarks = np.stack(landmark_array)
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

        faces = np.stack(face_array).transpose(0,3,1,2)
        tl = avg_rect.tl_corner()
        br = avg_rect.br_corner()
        return faces, landmarks, (tl.x, tl.y, br.x, br.y)

    def replace_landmark_boundary(self, landmarks):
        landmarks = copy.deepcopy(landmarks)
        arr = []
        for landmark in landmarks:
            landmark[:27] = self.__standard_landmark_mean[:27]
            arr.append(landmark)
        return np.stack(arr)

class RawFaceData(object):
    def __init__(
        self,
        video_list_pkl = './preprocessed/list_0.pkl',
    ):
        self.__processor = RawFaceDataProcessor()
        self.__video_list_pkl = video_list_pkl
        self.__list = None
        with open(video_list_pkl, 'rb') as fd:
            self.__list = load(fd)

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

    def __produce_file(self):
        with open(self.__video_list_pkl, 'wb') as fd:
            dump(self.__list, fd)

    def run(self):
        cnt = 1
        for utterance, utterance_map in tqdm(self.__list.items()):
            for train_val_test, code_map in utterance_map.items():
                for code, video_metadata in code_map.items():
                    video_path = video_metadata['path']
                    ltrb = video_metadata['ltrb']
                    landmarks = video_metadata['landmark']

                    if ltrb is not None and landmarks is not None and len(landmarks) == 29:
                        print(f"{video_path} have been processed, by pass")
                        continue
                    try:
                        frames = self.__video_frames(video_path)
                        if frames is None:
                            print("invalid video: {}".format(video_path))
                            continue
                        avg_rect = None
                        if ltrb is not None:
                            avg_rect = dlib.rectangle(*ltrb)
                        _, landmarks, ltrb = self.__processor.crop_face(frames, avg_rect)
                        landmarks = self.__processor.align_nose_mouth(landmarks)
                        landmarks = self.__processor.replace_landmark_boundary(landmarks)
                        video_metadata['ltrb'] = ltrb
                        video_metadata['landmark'] = landmarks
                    except Exception:
                        traceback.print_exc(file=sys.stdout)
                    cnt += 1
                    if cnt % 1000 == 0:
                        print(f"count = {cnt}, backup")
                        self.__produce_file()
        print("done")
        self.__produce_file()


def main():
    list_path = sys.argv[1]
    print(f"data from path: {list_path}")
    d = RawFaceData(video_list_pkl = list_path)
    d.run()

if __name__ == "__main__":
    main()
