#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
CKPT=../ckpt/delta_ckpt/nextgpt/7b_tiva_v0

# Level 1
# python eval.py --nextgpt_ckpt_path $CKPT --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LAQA --freeze_lm
# python eval.py --nextgpt_ckpt_path $CKPT --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LIQA --freeze_lm
# python eval.py --nextgpt_ckpt_path $CKPT --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LVQA --freeze_lm

# Level 2
# python eval.py --nextgpt_ckpt_path $CKPT --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_2/MAIC --freeze_lm
# python eval.py --nextgpt_ckpt_path $CKPT --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_2/MVIC --freeze_lm

# Level 3
python eval.py --nextgpt_ckpt_path $CKPT --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVH --freeze_lm
python eval.py --nextgpt_ckpt_path $CKPT --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVL --freeze_lm
# python eval.py --nextgpt_ckpt_path $CKPT --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVM --freeze_lm
# python eval.py --nextgpt_ckpt_path $CKPT --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVR --freeze_lm
python eval.py --nextgpt_ckpt_path $CKPT --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/VAH --freeze_lm
# python eval.py --nextgpt_ckpt_path $CKPT --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/VAR --freeze_lm

# Level 4
python eval.py --nextgpt_ckpt_path $CKPT --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVC --freeze_lm
python eval.py --nextgpt_ckpt_path $CKPT --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVLG --freeze_lm
python eval.py --nextgpt_ckpt_path $CKPT --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVQA --freeze_lm

# Level 5
python eval.py --nextgpt_ckpt_path $CKPT --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_5/AVLG --freeze_lm
python eval.py --nextgpt_ckpt_path $CKPT --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_5/AVQA --freeze_lm

# nohup bash scripts/eval.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_nextgpt_gpu3_$(date +%Y%m%d%H%M%S).log 2>&1 &