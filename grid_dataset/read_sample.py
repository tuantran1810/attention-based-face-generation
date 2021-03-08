from compress_pickle import load
import matplotlib.pyplot as plt
import numpy as np

def main():
    mp = load("/media/tuantran/raid-data/dataset/GRID/face_images_128_128/s8/pgwe1p.gzip", compression = 'gzip', set_default_extension = False)
    faces = mp['faces']
    mfcc = mp['mfcc']
    landmarks = mp['landmarks']
    print(mfcc.shape)
    print(landmarks.shape)
    print(faces.shape)
    # plt.figure()
    # plt.imshow(faces.transpose(1,2,0))
    # plt.show()

if __name__ == "__main__":
    main()
