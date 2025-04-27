#!/usr/bin/env python3
"""
inference_demo.py: 本地推理脚本，支持 text、image、audio、video 等多模态组合。
"""

import argparse
import os
import torch
from model.anyToImageVideoAudio import NextGPTModel
from config import load_config
from PIL import Image
import scipy.io.wavfile as wavfile
import imageio

def load_model(args):
    """
    加载并初始化 NextGPTModel，确保 cfg 中包含所有命令行参数和 YAML 配置。
    """
    # 1. 以命令行参数为基础
    cfg = vars(args).copy()

    # 2. 读取并合并 YAML 中的默认配置
    yaml_cfg = load_config(cfg)
    cfg.update(yaml_cfg)

    # 3. 确保 stage 和 nextgpt_ckpt_path 在 cfg 中
    # （通常已经包含，但下面做一次显式赋值以防万一）
    cfg['stage'] = args.stage
    cfg['nextgpt_ckpt_path'] = args.nextgpt_ckpt_path

    # 4. 创建模型
    model = NextGPTModel(**cfg)

    # 5. 加载 delta checkpoint
    delta = torch.load(
        os.path.join(args.nextgpt_ckpt_path, 'pytorch_model.pt'),
        map_location='cuda'
    )
    model.load_state_dict(delta, strict=False)
    return model.eval().half().cuda()

def extract_audio_from_video(video_path, output_wav):
    """
    可选：从 video 中抽取音轨并保存为 WAV。
    """
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()['fps']
    # 这里只是示例，真实可用 ffmpeg 或 moviepy 完成
    # 此处省略音频抽取具体实现
    return output_wav

def parse_args():
    parser = argparse.ArgumentParser(description='NextGPT 本地推理脚本')
    # --- 模型与 checkpoint ---
    parser.add_argument('--nextgpt_ckpt_path', type=str, required=True)
    parser.add_argument('--stage', type=int, default=3)
    parser.add_argument('--freeze_lm', type=bool, default=True)
    # --- 基础输入 ---
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--audio', type=str, default=None)
    parser.add_argument('--video', type=str, default=None)
    parser.add_argument('--use_video_audio', action='store_true')
    # --- 文本生成参数 ---
    parser.add_argument('--top_p', type=float, default=0.01)
    parser.add_argument('--temperature', type=float, default=1.0)
    # --- 图像生成参数 ---
    parser.add_argument('--guidance_scale_for_img', type=float, default=7.5)
    parser.add_argument('--num_inference_steps_for_img', type=int, default=50)
    parser.add_argument('--max_num_imgs', type=int, default=1)
    # --- 视频生成参数 ---
    parser.add_argument('--guidance_scale_for_vid', type=float, default=7.5)
    parser.add_argument('--num_inference_steps_for_vid', type=int, default=50)
    parser.add_argument('--max_num_vids', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--height', type=int, default=320)
    parser.add_argument('--width', type=int, default=576)
    parser.add_argument('--num_frames', type=int, default=24)
    # --- 音频生成参数 ---
    parser.add_argument('--guidance_scale_for_aud', type=float, default=7.5)
    parser.add_argument('--num_inference_steps_for_aud', type=int, default=50)
    parser.add_argument('--max_num_auds', type=int, default=1)
    parser.add_argument('--audio_length_in_s', type=int, default=9)
    # --- 其他超参 ---
    parser.add_argument('--filter_value', type=float, default=-float('Inf'))
    parser.add_argument('--min_word_tokens', type=int, default=10)
    parser.add_argument('--gen_scale_factor', type=float, default=4.0)
    parser.add_argument('--stops_id', nargs='+', type=int, default=[835])
    parser.add_argument('--load_sd', action='store_true', help='是否加载 SD 模态嵌入')
    parser.add_argument('--encounters', type=int, default=1)
    return parser.parse_args()

def run_inference(model, args):
    # 构造输入列表
    image_paths = [args.image] if args.image else []
    audio_paths = []
    if args.audio:
        audio_paths.append(args.audio)
    if args.video and args.use_video_audio:
        audio_paths.append(args.video)
    video_paths = [args.video] if args.video else []

    inputs = {
        'prompt': args.text,
        'image_paths':      image_paths,
        'audio_paths':      audio_paths,
        'video_paths':      video_paths,
        'top_p':            args.top_p,
        'temperature':      args.temperature,
        'max_tgt_len':      args.max_length,
        # 文本/多模态生成控制
        'filter_value':     args.filter_value,
        'min_word_tokens':  args.min_word_tokens,
        'gen_scale_factor': args.gen_scale_factor,
        'stops_id':         [args.stops_id],  # e.g. [[835]]
        'load_sd':          args.load_sd,
        # 随机种子生成器
        'generator':        torch.Generator(device='cuda').manual_seed(13),
        # 图像生成
        'guidance_scale_for_img':     args.guidance_scale_for_img,
        'num_inference_steps_for_img':args.num_inference_steps_for_img,
        'max_num_imgs':               args.max_num_imgs,
        # 视频生成
        'guidance_scale_for_vid':     args.guidance_scale_for_vid,
        'num_inference_steps_for_vid':args.num_inference_steps_for_vid,
        'max_num_vids':               args.max_num_vids,
        'height':                     args.height,
        'width':                      args.width,
        'num_frames':                 args.num_frames,
        # 音频生成
        'guidance_scale_for_aud':     args.guidance_scale_for_aud,
        'num_inference_steps_for_aud':args.num_inference_steps_for_aud,
        'max_num_auds':               args.max_num_auds,
        'audio_length_in_s':          args.audio_length_in_s,
        # 其他
        'ENCOUNTERS':                 args.encounters,
    }

    outputs = model.generate(inputs)
    print("=== Model Output ===")
    print(outputs)

if __name__ == '__main__':
    args  = parse_args()
    model = load_model(args)
    run_inference(model, args)