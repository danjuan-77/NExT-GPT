#!/bin/bash

CKPT=../ckpt/delta_ckpt/nextgpt/7b_tiva_v0

# 纯文本推理
python infer_demo.py \
  --nextgpt_ckpt_path $CKPT \
  --prompt "Hello" \
  --freeze_lm

# 文本 + 图片
python infer_demo.py \
  --nextgpt_ckpt_path $CKPT \
  --prompt "Describe the image." \
  --image_path ./asset/761183272.jpg \
  --freeze_lm

# 文本 + 音频
python infer_demo.py \
  --nextgpt_ckpt_path $CKPT \
  --prompt "Describe the audio." \
  --audio_path ./asset/1272-128104-0000.flac \
  --freeze_lm

# 文本 + 视频（仅画面，不使用视频自带音频）
python infer_demo.py \
  --nextgpt_ckpt_path $CKPT \
  --prompt "Describe the video." \
  --video_path ./asset/4405327307.mp4 \
  --freeze_lm

# 文本 + 视频（使用视频自带音频）
python infer_demo.py \
  --nextgpt_ckpt_path $CKPT \
  --prompt "Describe the video with its audio." \
  --audio_path ./asset/4405327307.wav \
  --video_path ./asset/4405327307.mp4 \
  --freeze_lm