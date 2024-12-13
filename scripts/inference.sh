#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python EAT/inference/inference2.py  \
    --source_path='/data/lmorove1/hwang258/Speech-Captioning-Dataset/audios/' \  # change this
    --save_path="/data/lmorove1/hwang258/Speech-Captioning-Dataset/audio_tagging.txt" \ # change this
    --label_file='EAT/inference/labels.csv' \ # don't change
    --model_dir='EAT' \ # don't change
    --checkpoint_dir='/data/lmorove1/hwang258/Speech-Captioning-Dataset/fairseq/EAT/EAT-large_epoch20_ft.pt' \ # change this
    --target_length=1024 \ # don't change
    --top_k_prediction=15 # don't change

# For optimal performance, 1024 is recommended for 10-second audio clips. (128 for 1-second)
# However, you should adjust the target_length parameter based on the duration and characteristics of your specific audio inputs.
# EAT-finetuned could make inference well even given truncated audio clips.
