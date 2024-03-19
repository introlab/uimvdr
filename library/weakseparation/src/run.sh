#!/bin/bash

source activate weak

# Bark supervised
python main.py --example --log --predict --num_of_iteration_test 10 --ckpt_path /home/jacob/dev/weakseparation/logs/mc-weak-separation/9wb9io4f

# Bark unsupervised
python main.py --example --log --predict --num_of_iteration_test 10 --ckpt_path /home/jacob/dev/weakseparation/logs/mc-weak-separation/31ypjv7l

# Bark unsupervised weigthing
python main.py --example --log --predict --num_of_iteration_test 10 --ckpt_path /home/jacob/dev/weakseparation/logs/mc-weak-separation/hs56ya47

# Bark Audioset
python main.py --example --log --predict --num_of_iteration_test 10 --ckpt_path /home/jacob/dev/weakseparation/logs/mc-weak-separation/bt8rv34b

# Bark Audioset weigthing
python main.py --example --log --predict --num_of_iteration_test 10 --ckpt_path /home/jacob/dev/weakseparation/logs/mc-weak-separation/qduqlu78

# # Speech supervised
# python main.py --example --log --predict --num_of_iteration_test 10 --ckpt_path /home/jacob/dev/weakseparation/logs/mc-weak-separation/oxf5w43t

# # Speech unsupervised
# python main.py --example --log --predict --num_of_iteration_test 10 --ckpt_path /home/jacob/dev/weakseparation/logs/mc-weak-separation/4do5uglh

# # Speech unsupervised weighting
# python main.py --example --log --predict --num_of_iteration_test 10 --ckpt_path /home/jacob/dev/weakseparation/logs/mc-weak-separation/yxwdeefy

# # Speech Audioset
# python main.py --example --log --predict --num_of_iteration_test 10 --ckpt_path /home/jacob/dev/weakseparation/logs/mc-weak-separation/o5u1z7wa

# # Speech Audioset weighting
# python main.py --example --log --predict --num_of_iteration_test 10 --ckpt_path /home/jacob/dev/weakseparation/logs/mc-weak-separation/ng1nh4wp

conda deactivate