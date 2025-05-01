#!/bin/bash

CKPT=../ckpt/delta_ckpt/nextgpt/7b_tiva_v0


python eval.py \
    --nextgpt_ckpt_path $CKPT \
    --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LAQA \
    --freeze_lm