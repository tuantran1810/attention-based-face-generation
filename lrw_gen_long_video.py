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
    audio_source_folder = "/media/tuantran/raid-data/dataset/LRW/attention-based-face-generation/plain_audio"
    output_path = "./lrw_validator_output"
    path = sys.argv[1]
    mp = None
    with open(path, 'rb') as fd:
        mp = pickle.load(fd)
        print(mp.keys())

    utterances = mp['utterance']
    codes = mp['code']
    datasets = mp['dataset']
    original_images = mp['original_images']
    attention_maps = mp['attention_map']
    color_images = mp['color_images']
    fake_images = mp['fake_images']

    final_images = []
    final_audio = []

    for i in range(len(utterances)):
        utterance, code, dataset = utterances[i], codes[i], datasets[i]
        original_image, attention_map, color_image, fake_image = original_images[i], attention_maps[i], color_images[i], fake_images[i]
        fake_image = fake_image.transpose(1,2,3,0)
        original_image = original_image.transpose(1,2,3,0)
        color_image = color_image.transpose(1,2,3,0)
        attention_map = (np.repeat(attention_map.transpose(1,2,3,0), 3, axis = 3) * 255.0).astype(np.uint8)

        image = np.concatenate(
            [
                original_image,
                fake_image,
                color_image,
                attention_map,
            ],
            axis = 2,
        )

        for j in range(23):
            tmp = cv2.cvtColor(image[j], cv2.COLOR_BGR2RGB)
            final_images.append(tmp)

        audio_src_file = os.path.join(audio_source_folder, utterance, dataset, code + '.wav')
        audio, sr = librosa.load(audio_src_file, sr = 8000)
        samples = int(sr * 1.16)
        stride = samples//29
        window = stride * 23
        offset = 3 * stride
        audio = audio[offset:offset+window]
        final_audio.append(audio)

    final_images = np.stack(final_images, axis=0)
    plain_video_path = os.path.join(output_path, 'plain_video.mp4')
    vidwrite(plain_video_path, final_images, vcodec='libx264', fps=25)

    plain_audio_path = os.path.join(output_path, 'plain_audio.wav')
    final_audio = np.concatenate(final_audio)
    wav.write(plain_audio_path, 8000, final_audio) 

    final_video_path = os.path.join(output_path, 'final.mp4')
    v = ffmpeg.input(plain_video_path)
    a = ffmpeg.input(plain_audio_path)
    final = ffmpeg.output(v['v'], a['a'], final_video_path, loglevel="panic")
    final.run()


if __name__ == "__main__":
    main()
