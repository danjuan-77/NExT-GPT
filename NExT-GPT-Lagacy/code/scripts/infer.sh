#!/bin/bash

# 纯文本推理
python infer_demo.py \
  --nextgpt_ckpt_path ../ckpt/delta_ckpt/nextgpt/7b_tiva_v0 \
  --text "Hello" \
  --freeze_lm True

# 文本 + 图片
python infer_demo.py \
  --nextgpt_ckpt_path ../ckpt/delta_ckpt/nextgpt/7b_tiva_v0 \
  --text "Describe the image." \
  --image ./asset/761183272.jpg \
  --freeze_lm True

# 文本 + 音频
python infer_demo.py \
  --nextgpt_ckpt_path ../ckpt/delta_ckpt/nextgpt/7b_tiva_v0 \
  --text "Describe the audio." \
  --audio ./asset/1272-128104-0000.flac \
  --freeze_lm True

# 文本 + 视频（不使用视频音频）
python infer_demo.py \
  --nextgpt_ckpt_path ../ckpt/delta_ckpt/nextgpt/7b_tiva_v0 \
  --text "Describe the video." \
  --video ./asset/4405327307.mp4 \
  --freeze_lm True

# 文本 + 视频（使用视频自带音频）
python infer_demo.py \
  --nextgpt_ckpt_path ../ckpt/delta_ckpt/nextgpt/7b_tiva_v0 \
  --text "Describe the video with its audio" \
  --video ./asset/4405327307.mp4 \
  --use_video_audio \
  --freeze_lm True