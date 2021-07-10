import os, random, pickle

def main():
    image_root_path = "/media/tuantran/rapid-data/dataset/GRID/face_images_128"
    random.seed(0)
    paths = list()

    for path, _ , files in os.walk(image_root_path):
        identity = path.split('/')[-1]
        for name in files:
            images_path = identity + '/' + name
            name = name.split('.')
            if name[-1] != 'gzip': continue
            name = name[0]
            paths.append((identity, name))

    random.shuffle(paths)
    print(paths)
    print(f"num of videos: {len(paths)}")
    n_train = int(0.9*len(paths))
    n_remain = len(paths) - n_train
    n_val = n_remain//2
    train_set = paths[0:n_train]
    val_set = paths[n_train:(n_train+n_val)]
    test_set = paths[(n_train+n_val):]

    print(f"train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")

    out = {
        "train": train_set,
        "val": val_set,
        "test": test_set,
    }

    with open("./preprocessed/dataset_split.pkl", 'wb') as fd:
        pickle.dump(out, fd)

if __name__ == "__main__":
    main()
