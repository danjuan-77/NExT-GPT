#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
CKPT=../ckpt/delta_ckpt/nextgpt/7b_tiva_v0

python eval.py --nextgpt_ckpt_path $CKPT --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVL --freeze_lm

# nohup bash scripts/eval_gpu1.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_nextgpt_unimodal_gpu1_$(date +%Y%m%d%H%M%S).log 2>&1 &