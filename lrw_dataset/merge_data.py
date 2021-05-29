import os, sys, pickle

if __name__ == "__main__":
    input_path = sys.argv[1]
    output_name = sys.argv[2]
    key = None
    if len(sys.argv) == 4:
        key = sys.argv[3]
    output_path = os.path.join(input_path, output_name)
    final = {}
    for path, _ , files in os.walk(input_path):
        for file in files:
            if file.split('.')[-1] != 'pkl':
                continue
            landmark_map = None
            with open(os.path.join(path, file), 'rb') as fd:
                landmark_map = pickle.load(fd)
            for u, umap in landmark_map.items():
                if u not in final: final[u] = dict()
                for d, dmap in umap.items():
                    if d not in final[u]: final[u][d] = dict()
                    for c, data in dmap.items():
                        if data is None:
                            continue
                        if key is None or key == "":
                            final[u][d][c] = data
                        else:
                            final[u][d][c] = data[key]
    with open(output_path, 'wb') as fd:
        pickle.dump(final, fd)
