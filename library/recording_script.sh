#!/bin/bash

echo "Starting recording"

for path in /home/jacob/Dev/weakseparation/library/dataset/data_to_play/*.wav; do
	[ -e "$path" ] || continue
	file_name=$(basename $path)
	echo "RECORDING: $file_name"

	# Start recording on PIs and save pid
	#----------- ReSpeaker -----------------
	ssh -n -f jacobpi@10.0.0.126 "./record.sh $file_name"
	scp jacobpi@10.0.0.126:~/dev/dataset/save_pid.txt ~/Dev/weakseparation/library/dataset/custom/ReSpeaker
	ssh -n -f jacobpi@10.0.0.126 "rm ~/dev/dataset/save_pid.txt"
	#----------- Kinect -----------------
	ssh -n -f francoisgrondin1@10.0.0.224 "./record.sh $file_name"
	scp francoisgrondin1@10.0.0.224:~/dev/dataset/save_pid.txt ~/Dev/weakseparation/library/dataset/custom/Kinect
	ssh -n -f francoisgrondin1@10.0.0.224 "rm ~/dev/dataset/save_pid.txt"
	#----------- 16Sounds -----------------
	ssh -n -f francoisgrondin2@10.0.0.22 "./record.sh $file_name"
	scp francoisgrondin2@10.0.0.22:~/dev/dataset/save_pid.txt ~/Dev/weakseparation/library/dataset/custom/16Sounds
	ssh -n -f francoisgrondin2@10.0.0.22 "rm ~/dev/dataset/save_pid.txt"

	# Play and this PC
	aplay "$path"

	# Stop recording on PIs and cleanup
	#----------- ReSpeaker -----------------
	ssh -n -f jacobpi@10.0.0.126 "kill "$(cat ~/Dev/weakseparation/library/dataset/custom/ReSpeaker/save_pid.txt)""
	rm ~/Dev/weakseparation/library/dataset/custom/ReSpeaker/save_pid.txt
	
	#----------- Kinect -----------------
	ssh -n -f francoisgrondin1@10.0.0.224 "kill "$(cat ~/Dev/weakseparation/library/dataset/custom/Kinect/save_pid.txt)""
	rm ~/Dev/weakseparation/library/dataset/custom/Kinect/save_pid.txt
	
	#----------- 16Sounds -----------------
	ssh -n -f francoisgrondin2@10.0.0.22 "kill "$(cat ~/Dev/weakseparation/library/dataset/custom/16Sounds/save_pid.txt)""
	rm ~/Dev/weakseparation/library/dataset/custom/16Sounds/save_pid.txt
	
	# Copy files
	scp jacobpi@10.0.0.126:~/dev/dataset/$file_name ~/Dev/weakseparation/library/dataset/custom/ReSpeaker
	scp francoisgrondin1@10.0.0.224:~/dev/dataset/$file_name ~/Dev/weakseparation/library/dataset/custom/Kinect
	scp francoisgrondin2@10.0.0.22:~/dev/dataset/$file_name ~/Dev/weakseparation/library/dataset/custom/16Sounds
done

echo "Done recording"

#francoisgrondin1@10.0.0.224 : kinect
#jacobpi@10.0.0.126 : respeaker
#francoisgrondin2@10.0.0.22 : 16Sounds
