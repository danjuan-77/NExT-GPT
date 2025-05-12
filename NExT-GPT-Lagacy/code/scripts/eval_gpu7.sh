#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
CKPT=../ckpt/delta_ckpt/nextgpt/7b_tiva_v0

python eval.py --nextgpt_ckpt_path $CKPT --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_5/AVQA --freeze_lm

# nohup bash scripts/eval_gpu7.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_nextgpt_unimodal_gpu7_$(date +%Y%m%d%H%M%S).log 2>&1 &