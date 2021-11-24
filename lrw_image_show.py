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

    for i in range(len(utterances)):
        utterance, code, dataset = utterances[i], codes[i], datasets[i]
        file_name_id = '{}_{}'.format(utterance, code)
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
            axis = 1,
        )

        # figimg, axesimg = plt.subplots(8, 3)
        for j in range(23):
            image[j] = cv2.cvtColor(image[j], cv2.COLOR_BGR2RGB)

        image = np.concatenate(image, axis=1)
        plt.figure(frameon=False)
        plt.imshow(image)
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    main()
