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
        for j in range(16):
            image[j] = cv2.cvtColor(image[j], cv2.COLOR_BGR2RGB)

        image = np.concatenate(image, axis=1)
        plt.figure(frameon=False)
        plt.imshow(image)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
