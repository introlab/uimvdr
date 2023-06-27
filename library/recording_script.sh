#!/bin/bash

echo "Starting recording"

for path in /home/jacob/Dev/weakseparation/library/dataset/data_to_play/*.wav; do
	[ -e "$path" ] || continue
	file_name=$(basename $path)
	echo "RECORDING: $file_name"

	# Start recording on PIs and save pid
	#----------- ReSpeaker -----------------
	ssh -n -f jacobpi@10.0.0.126 "./record.sh $file_name"
	scp jacobpi@10.0.0.126:~/dev/dataset/save_pid_respeaker.txt ~/Dev/weakseparation/library/dataset/custom
	ssh -n -f jacobpi@10.0.0.126 "rm ~/dev/dataset/save_pid_respeaker.txt"
	#----------- Kinect -----------------
	ssh -n -f francoisgrondin1@10.0.0.224 "./record.sh $file_name"
	scp francoisgrondin1@10.0.0.224:~/dev/dataset/save_pid_kinect.txt ~/Dev/weakseparation/library/dataset/custom
	ssh -n -f francoisgrondin1@10.0.0.224 "rm ~/dev/dataset/save_pid_kinect.txt"
	#----------- 16Sounds -----------------
	ssh -n -f francoisgrondin2@10.0.0.22 "./record.sh $file_name"
	scp francoisgrondin2@10.0.0.22:~/dev/dataset/save_pid_16sounds.txt ~/Dev/weakseparation/library/dataset/custom
	ssh -n -f francoisgrondin2@10.0.0.22 "rm ~/dev/dataset/save_pid_16sounds.txt"

	# Play and this PC
	aplay "$path"

	# Stop recording on PIs and cleanup
	#----------- ReSpeaker -----------------
	ssh -n -f jacobpi@10.0.0.126 "kill "$(cat ~/Dev/weakseparation/library/dataset/custom/save_pid_respeaker.txt)""
	rm ~/Dev/weakseparation/library/dataset/custom/save_pid_respeaker.txt
	
	#----------- Kinect -----------------
	ssh -n -f francoisgrondin1@10.0.0.224 "kill "$(cat ~/Dev/weakseparation/library/dataset/custom/save_pid_kinect.txt)""
	rm ~/Dev/weakseparation/library/dataset/custom/save_pid_kinect.txt
	
	#----------- 16Sounds -----------------
	ssh -n -f francoisgrondin2@10.0.0.22 "kill "$(cat ~/Dev/weakseparation/library/dataset/custom/save_pid_16sounds.txt)""
	rm ~/Dev/weakseparation/library/dataset/custom/save_pid_16sounds.txt
	
	# Copy files
	mkdir -p ~/Dev/weakseparation/library/dataset/custom/$1/ReSpeaker/$2
	mkdir -p ~/Dev/weakseparation/library/dataset/custom/$1/Kinect/$2
	mkdir -p ~/Dev/weakseparation/library/dataset/custom/$1/16Sounds/$2

	scp jacobpi@10.0.0.126:~/dev/dataset/$file_name ~/Dev/weakseparation/library/dataset/custom/$1/ReSpeaker/$2
	scp francoisgrondin1@10.0.0.224:~/dev/dataset/$file_name ~/Dev/weakseparation/library/dataset/custom/$1/Kinect/$2
	scp francoisgrondin2@10.0.0.22:~/dev/dataset/$file_name ~/Dev/weakseparation/library/dataset/custom/$1/16Sounds/$2
done

echo "Done recording"

#francoisgrondin1@10.0.0.224 : kinect
#jacobpi@10.0.0.126 : respeaker
#francoisgrondin2@10.0.0.22 : 16Sounds
