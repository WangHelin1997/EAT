#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python EAT/inference/inference2.py  \
    --source_path='/data/lmorove1/hwang258/Speech-Captioning-Dataset/audios/' \
    --save_path="/data/lmorove1/hwang258/Speech-Captioning-Dataset/audio_tagging.txt" \
    --label_file='EAT/inference/labels.csv' \
    --model_dir='EAT' \
    --checkpoint_dir='/data/lmorove1/hwang258/Speech-Captioning-Dataset/fairseq/EAT/EAT-large_epoch20_ft.pt' \
    --target_length=1024 \
    --top_k_prediction=15

# For optimal performance, 1024 is recommended for 10-second audio clips. (128 for 1-second)
# However, you should adjust the target_length parameter based on the duration and characteristics of your specific audio inputs.
# EAT-finetuned could make inference well even given truncated audio clips.
