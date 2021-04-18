import sys, os
sys.path.append(os.path.dirname(__file__))

import cv2
import ffmpeg
import librosa
import numpy as np
from pathlib import Path
import pickle
from matplotlib import pyplot as plt
from utils.media import vidwrite
import scipy.io.wavfile as wav

def main():
    audio_source_folder = "/media/tuantran/raid-data/dataset/GRID/audio_50"
    path = sys.argv[1]
    mp = None
    with open(path, 'rb') as fd:
        mp = pickle.load(fd)
        print(mp.keys())
    folder = path.split('.')[0] + '_output'
    images_folder = os.path.join(folder, 'images')
    videos_folder = os.path.join(folder, 'videos')
    audios_folder = os.path.join(folder, 'audios')
    final_folder = os.path.join(folder, 'final')
    Path(images_folder).mkdir(parents=True, exist_ok=True)
    Path(videos_folder).mkdir(parents=True, exist_ok=True)
    Path(audios_folder).mkdir(parents=True, exist_ok=True)
    Path(final_folder).mkdir(parents=True, exist_ok=True)

    identities = mp['identity']
    codes = mp['code']
    data_start_indexes = mp['data_start_index']
    original_images = mp['original_images']
    attention_maps = mp['attention_map']
    color_images = mp['color_images']
    fake_images = mp['fake_images']

    for i in range(len(identities)):
        identity, code, data_start_index = identities[i], codes[i], data_start_indexes[i]
        file_name_id = '{}_{}_{}'.format(identity, code, data_start_index)
        original_image, attention_map, color_image, fake_image = original_images[i], attention_maps[i], color_images[i], fake_images[i]
        fake_image = fake_image.transpose(1,2,3,0)
        original_image = original_image.transpose(1,2,3,0)
        color_image = color_image.transpose(1,2,3,0)
        attention_map = (np.repeat(attention_map.transpose(1,2,3,0), 3, axis = 3) * 255.0).astype(np.uint8)

        image = np.concatenate(
            [
                fake_image,
                original_image,
                color_image,
                attention_map,
            ],
            axis = 2,
        )

        figimg, axesimg = plt.subplots(4, 4)
        for j in range(16):
            image[j] = cv2.cvtColor(image[j], cv2.COLOR_BGR2RGB)

        for j in range(16):
            row, col = j//4, j%4
            axesimg[row][col].imshow(image[j])
            axesimg[row][col].axis('off')
        figimg.tight_layout()
        figimg.set_figheight(5)
        figimg.set_figwidth(20)
        figimg.savefig(os.path.join(images_folder, file_name_id+'.png'), dpi = 100)

        plain_video_path = os.path.join(videos_folder, file_name_id+'.mp4')
        vidwrite(plain_video_path, image, vcodec='libx264', fps=25)

        audio_src_file = os.path.join(audio_source_folder, identity, code + '.wav')
        audio, sr = librosa.load(audio_src_file, sr = 50000)
        samples = sr * 3
        stride = samples//75
        window = stride * 16
        offset = data_start_index * stride
        audio = audio[offset:offset+window]
        audio_path = os.path.join(audios_folder, file_name_id+'.wav')
        wav.write(audio_path, sr, audio) 

        final_video_path = os.path.join(final_folder, file_name_id+'.mp4')
        v = ffmpeg.input(plain_video_path)
        a = ffmpeg.input(audio_path)
        final = ffmpeg.output(v['v'], a['a'], final_video_path, loglevel="panic")
        final.run()


if __name__ == "__main__":
    main()
