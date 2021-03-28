# Data preprocessing process:
- Run raw_face_data.py first for cropping face images and package them into gzip pickle files
- Run raw_landmark_data.py to produce raw landmarks
- Run landmark_standardize.py to standardize landmarks
- Run landmark_pca.py to find pca feature of the standard landmark dataset
- Run mfcc_data.py for producing mfcc audio features
