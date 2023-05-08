#!/bin/bash

echo "Starting recording"

for path in /home/jacob/Dev/weakseparation/library/dataset/data_to_play/*.wav; do
	[ -e "$path" ] || continue
	file_name=$(basename $path)
	echo $file_name
	ssh -n -f racecar@10.0.0.126 "nohup arecord -D plughw:1,0 -r 16000 -c 6 -f S16_LE -d 4 $file_name > /dev/null 2>&1 &"
	sleep 1
	aplay "$path"
	sleep 1
	scp racecar@10.0.0.126:~/$file_name ~/Dev/weakseparation/library/dataset/custom
done

echo "Done recording"
