from re import U
import sys, os, cv2, traceback
sys.path.append(os.path.dirname(__file__))
import numpy as np
from pickle import load
from tqdm import tqdm
from pathlib import Path

class VideoAudioSplitter(object):
    def __init__(
        self,
        video_list_pkl = './preprocessed/list_0.pkl',
        video_output_folder = './preprocessed/plain_video',
        audio_output_folder = './preprocessed/plain_audio',
    ):
        self.__video_list_pkl = video_list_pkl
        self.__video_output_folder = video_output_folder
        self.__audio_output_folder = audio_output_folder
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

    def __resize_batch(self, frames, size):
        tmp_frames = []
        for frame in frames:
            frame = cv2.resize(frame, size)
            frame = np.expand_dims(frame, 0)
            tmp_frames.append(frame)
        frames = np.concatenate(tmp_frames, axis = 0)
        return frames

    def __write_video(self, video, path, fps = 25):
        f, w, h, _ = video.shape
        print(video.shape)
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
        for i in range(f):
            data = video[i]
            out.write(data)
        out.release()

    def run(self):
        for utterance, utterance_map in tqdm(self.__list.items()):
            audio_utterance_path = os.path.join(self.__audio_output_folder, utterance)
            video_utterance_path = os.path.join(self.__video_output_folder, utterance)
            Path(audio_utterance_path).mkdir(parents = True, exist_ok = True)
            Path(video_utterance_path).mkdir(parents = True, exist_ok = True)
            # print(utterance)
            for train_val_test, code_map in utterance_map.items():
                audio_folder = os.path.join(audio_utterance_path, train_val_test)
                video_folder = os.path.join(video_utterance_path, train_val_test)
                Path(audio_folder).mkdir(parents = True, exist_ok = True)
                Path(video_folder).mkdir(parents = True, exist_ok = True)
                for code, video_metadata in code_map.items():
                    video_path = video_metadata['path']
                    ltrb = video_metadata['ltrb']
                    landmark = video_metadata['landmark']
                    if video_path is None or ltrb is None or landmark is None or len(ltrb) != 4 or landmark.shape != (29,68,2):
                        continue

                    audio_file_path = os.path.join(audio_folder, code + '.wav')
                    if not os.path.exists(audio_file_path):
                        audio_command = f'ffmpeg -i {video_path} -ar 8000 -ac 1 {audio_file_path}'
                        try:
                            os.system(audio_command)
                        except BaseException:
                            traceback.print_exc(file=sys.stdout)

                    video_file_path = os.path.join(video_folder, code + '.mp4')
                    if not os.path.exists(video_file_path):
                        frames = self.__video_frames(video_path)
                        l,t,r,b = ltrb
                        frames = frames[:,l:r,t:b,:]
                        frames = self.__resize_batch(frames, (128,128))
                        self.__write_video(frames, video_file_path)

        print("done")

def main():
    list_path = sys.argv[1]
    print(f"data from path: {list_path}")
    d = VideoAudioSplitter(
        video_list_pkl = list_path,
        video_output_folder = "/media/tuantran/raid-data/dataset/LRW/attention-based-face-generation/plain_video",
        audio_output_folder = "/media/tuantran/raid-data/dataset/LRW/attention-based-face-generation/plain_audio",
    )
    d.run()

if __name__ == "__main__":
    main()
