import os
from huggingface_hub import snapshot_download


snapshot_download(repo_id="lmsys/vicuna-7b-v1.5",repo_type="model",local_dir="pretrain_ckpt/vicuna-7b-v1.5")

snapshot_download(repo_id="ChocoWu/nextgpt_7b_tiva_v0",repo_type="model",local_dir="checkpoints/nextgpt-7b-v1.5")
