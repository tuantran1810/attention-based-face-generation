#!/bin/bash
for i in {0..5}
do
    python split_video_audio.py /media/tuantran/raid-data/dataset/LRW/attention-based-face-generation/list_data/list_$i.pkl &
    pids[${i}]=$!
done

for pid in ${pids[*]}; do
    wait $pid
done
