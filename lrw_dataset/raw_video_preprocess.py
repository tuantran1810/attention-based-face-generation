import cv2, sys, os, dlib
sys.path.append(os.path.dirname(__file__))
from os import path
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from pickle import dump
from compress_pickle import dump as cdump
from tqdm import tqdm

class RawFaceDataProcessor(object):
    def __init__(
        self,
        face_expected_ratio = "1:1",
        horizontal_landmark_scale = 2.2,
        device = 'cpu',
    ):
        self.__device = device
        fw, fh = face_expected_ratio.split(':')
        self.__faceratiowh = int(fw)/int(fh)

        self.__frontal_face_detector = dlib.get_frontal_face_detector()
        self.__landmark_detector = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
        self.__horizontal_landmark_scale = horizontal_landmark_scale


    def resize_batch(self, frames, size):
        tmp_frames = []
        for frame in frames:
            frame = cv2.resize(frame, size)
            frame = np.expand_dims(frame, 0)
            tmp_frames.append(frame)
        frames = np.concatenate(tmp_frames, axis = 0)
        return frames

    def __detect_landmark(self, frame, rect, dtype="int"):
        assert len(rect) >= 1
        shape = self.__landmark_detector(frame, rect[0])
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def crop_face(self, frames):
        frame_array = []
        rect_array = []
        for frame in frames:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_array.append(frame)
            rect = self.__frontal_face_detector(frame, 0)
            rect_array.append(rect)

        assert len(rect_array) == len(frame_array)
        for i, rect in enumerate(rect_array):
            frame = frame_array[i]
            landmark = self.__detect_landmark(frame, rect)
            for point in landmark:
                cv2.circle(frame, point, 2, (255,0,0), 2)
            # rect = rect[0].rect
            tl = rect[0].tl_corner()
            br = rect[0].br_corner()
            cv2.rectangle(frame, (tl.x, tl.y), (br.x, br.y), (0,255,0), 2)
            
            plt.figure()
            plt.imshow(frame)
            plt.show()
            plt.close()
        # frame_array.append(frame)
        # final = np.concatenate(frame_array)
        # print(final.shape)
        return frames


        # _, _, landmarks = self.__face_dectector.detect(frames, landmarks = True)

        # tmp_lmark = np.zeros((5, 2))
        # cnt = 0.0
        # for lm in landmarks:
        #     if lm is None:
        #         continue
        #     tmp_lmark += lm[0]
        #     cnt += 1.0
        # if cnt < 1:
        #     return None

        # tmp_lmark = tmp_lmark/cnt
        # left_eye, right_eye, nose, left_mouth, right_mouth = tmp_lmark
        # x_nose, y_nose = nose

        # avg_left = (left_eye + left_mouth) / 2.0
        # avg_right = (right_eye + right_mouth) / 2.0
        # d_nose_left, _ = nose - avg_left
        # d_nose_right, _ = avg_right - nose
        # x1 = int(x_nose - self.__horizontal_landmark_scale*d_nose_left)
        # x2 = int(x_nose + self.__horizontal_landmark_scale*d_nose_right)

        # dx = x2 - x1
        # dy = int(dx/self.__faceratiowh)

        # avg_eye = (left_eye + right_eye) / 2.0
        # avg_mouth = (left_mouth + right_mouth) / 2.0
        # _, d_eyes_nose = nose - avg_eye
        # _, d_nose_mouth = avg_mouth - nose
        # d_nose_y1 = int(dy*(d_eyes_nose/(d_eyes_nose + d_nose_mouth)))
        # d_nose_y2 = dy - d_nose_y1
        # y1 = int(y_nose - d_nose_y1)
        # y2 = int(y_nose + d_nose_y2)
        # frames = frames[:,y1:y2,x1:x2,:]
        # return frames

class RawFaceData(object):
    def __init__(
        self,
        videorootfolder = "./sample_videos",
        plain_videos_outputpath = "./preprocessed/plain_videos",
        video_ext = "mp4",
        face_expected_ratio = "1:1",
        horizontal_landmark_scale = 2.2,
        output_w = 128,
        device = 'cpu',
    ):
        '''
        self.__paths: map[identity]pathsList
        '''

        self.__device = device
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
                    videomap[code] = os.path.join(path, name)
            self.__paths[utterance][train_val_test] = videomap

        Path(plain_videos_outputpath).mkdir(parents = True, exist_ok = True)
        for path, _ , files in os.walk(plain_videos_outputpath):
            if len(files) == 0:
                continue

            segments = path.split('/')
            utterance = segments[-2]
            train_val_test = segments[-1]

            if utterance not in self.__paths or train_val_test not in self.__paths[utterance]:
                continue

            videomap = self.__path[utterance][train_val_test]
            for name in files:
                code, file_ext = name.split('.')
                if file_ext == video_ext:
                    if code in videomap:
                        print('{}/{}/{} existed'.format(utterance, train_val_test, code))
                        self.__paths[utterance][train_val_test].pop(code, None)
                        if len(self.__paths[utterance][train_val_test]) == 0:
                            self.__paths[utterance].pop(train_val_test, None)
                        if len(self.__paths[utterance]) == 0:
                            self.__paths.pop(utterance, None)

        self.__plain_videos_outputpath = plain_videos_outputpath
        fw, fh = face_expected_ratio.split(':')
        self.__faceratiowh = int(fw)/int(fh)
        self.__output_w = output_w
        self.__output_h = int(output_w/self.__faceratiowh)

        self.__horizontal_landmark_scale = horizontal_landmark_scale
        self.__processor = RawFaceDataProcessor(
            face_expected_ratio = face_expected_ratio,
            horizontal_landmark_scale = horizontal_landmark_scale,
            device = device,
        )

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

    def __produce_file(self, outputpath, faces, identity, code):
        image_mp = {}
        image_mp['identity'] = identity
        image_mp['code'] = code
        image_mp['faces'] = faces
        filename = '{}.gzip'.format(code)
        filepath = os.path.join(outputpath, identity, filename)
        with open(filepath, 'wb') as fd:
            cdump(image_mp, fd, compression = 'gzip')

    def run(self):
        for utterance, utterance_map in tqdm(self.__paths.items()):
            utterance_path = os.path.join(self.__plain_videos_outputpath, utterance)
            Path(utterance_path).mkdir(parents = True, exist_ok = True)
            for train_val_test, code_map in utterance_map.items():
                train_val_test_path = os.path.join(utterance_path, train_val_test)
                Path(train_val_test_path).mkdir(parents = True, exist_ok = True)
                for code, video_path in code_map.items():
                    frames = self.__video_frames(video_path)
                    if frames is None:
                        print("invalid video: {}".format(video_path))
                        continue
                    faces = self.__processor.crop_face(frames)
                    # plt.figure()
                    # plt.imshow(faces[10])
                    # plt.show()
                    # plt.close()
                    # if faces is None or faces.shape[0] != 75:
                    #     print("invalid faces or landmarks: {}".format(video_path))
                    #     continue
                    # faces = self.__processor.resize_batch(faces, (self.__output_w, self.__output_h)).transpose(0,3,1,2)
                    print(video_path)
                    # self.__produce_file(self.__image_outputpath, faces, identity, code)

def main():
    d = RawFaceData(
        # videorootfolder = "/media/tuantran/raid-data/dataset/GRID/video",
        # plain_videos_outputpath = "/media/tuantran/rapid-data/dataset/GRID/face_images_128",
        # device = 'cuda:0',
    )
    d.run()

if __name__ == "__main__":
    main()
