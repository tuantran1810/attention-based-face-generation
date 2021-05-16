#!/bin/bash
for i in {0..5}
do
    python raw_video_preprocess.py /media/tuantran/raid-data/dataset/LRW/attention-based-face-generation/list_data/list_$i.pkl &
    pids[${i}]=$!
done

for pid in ${pids[*]}; do
    wait $pid
done
