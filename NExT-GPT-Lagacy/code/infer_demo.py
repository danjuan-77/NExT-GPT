import os
import argparse
import tempfile
import torch
import imageio
import scipy.io
from PIL import Image
from model.anyToImageVideoAudio import NextGPTModel
from config import load_config


def save_image_to_local(image: Image.Image, output_dir: str) -> str:
    """
    Save a PIL Image to a temporary JPG file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, next(tempfile._get_candidate_names()) + '.jpg')
    image.save(path)
    return path


def save_video_to_local(frames, output_dir: str, fps: int = 8) -> str:
    """
    Save a list of frames to an MP4 video file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, next(tempfile._get_candidate_names()) + '.mp4')
    writer = imageio.get_writer(path, format='FFMPEG', fps=fps)
    for f in frames:
        writer.append_data(f)
    writer.close()
    return path


def save_audio_to_local(audio, output_dir: str, sample_rate: int = 16000) -> str:
    """
    Save a 1D torch or numpy audio array as a WAV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, next(tempfile._get_candidate_names()) + '.wav')
    data = audio.cpu().numpy() if hasattr(audio, 'cpu') else audio
    scipy.io.wavfile.write(path, rate=sample_rate, data=data)
    return path


def parse_response(model_outputs, output_dir: str):
    """
    Handle model outputs, saving any media locally.
    Returns:
      - text: concatenated strings
      - media_info: list of dicts with saved file paths
    """
    texts = []
    media = []
    for out in model_outputs:
        if isinstance(out, str):
            texts.append(out)
        elif isinstance(out, dict):
            entry = {}
            if 'img' in out:
                entry['images'] = [save_image_to_local(img, output_dir) for img in out['img'] if isinstance(img, Image.Image)]
            if 'vid' in out:
                entry['videos'] = [save_video_to_local(vid, output_dir) for vid in out['vid']]
            if 'aud' in out:
                entry['audios'] = [save_audio_to_local(aud, output_dir) for aud in out['aud']]
            media.append(entry)
    return "\n".join(texts), media


def build_prompt(prompt: str, image_path=None, audio_path=None, video_path=None, history=None) -> str:
    """
    Build the full prompt string with modality tags and optional chat history.
    Supports single path or list of paths for each modality.
    """
    history = history or []
    text = ''

    # Build history context
    if not history:
        text += '### Human: '
    else:
        for i, (q, a) in enumerate(history):
            sep = '###' if i == 0 else ''
            text += f'{sep} Human: {q}\n### Assistant: {a}\n'
        text += '### Human: '

    # Normalize to list
    def normalize(x):
        if not x:
            return []
        return x if isinstance(x, list) else [x]

    for img in normalize(image_path):
        text += f'<Image>{img}</Image> '

    for aud in normalize(audio_path):
        text += f'<Audio>{aud}</Audio> '

    for vid in normalize(video_path):
        text += f'<Video>{vid}</Video> '

    text += prompt
    print('Constructed prompt_text:', text)
    return text


def main():
    filter_value = -float('Inf')
    min_word_tokens = 10
    gen_scale_factor = 4.0
    stops_id = [[835]]
    ENCOUNTERS = 1
    load_sd = True
    generator = torch.Generator(device='cuda').manual_seed(13)

    max_num_imgs = 1
    max_num_vids = 1
    height = 320
    width = 576

    max_num_auds = 1
    max_length = 246
    parser = argparse.ArgumentParser(description="NExT-GPT CLI for multimodal inference")
    parser.add_argument('--model', type=str, default='nextgpt')
    parser.add_argument('--nextgpt_ckpt_path', type=str, required=True,
                        help='Path to model checkpoint directory')
    parser.add_argument('--stage', type=int, default=3,
                        help='Model stage for generation (must match training)')
    # Inference control
    parser.add_argument('--freeze_lm', action='store_true',
                        help='Freeze the LLM weights for inference only')

    # Inputs
    parser.add_argument('--prompt', type=str, default='', help='Text prompt')
    parser.add_argument('--image_path', type=str, default=None, help='Path to an input image')
    parser.add_argument('--audio_path', type=str, default=None, help='Path to an input audio file (wav)')
    parser.add_argument('--video_path', type=str, default=None, help='Path to an input video file')

    # Sampling options
    parser.add_argument('--top_p', type=float, default=0.01)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max_tgt_len', type=int, default=246)

    args = vars(parser.parse_args())
    args.update(load_config(args))

    # Initialize and load model
    model = NextGPTModel(**args)
    ckpt = torch.load(os.path.join(args['nextgpt_ckpt_path'], 'pytorch_model.pt'), map_location='cuda')
    model.load_state_dict(ckpt, strict=False)
    model.eval().half().cuda()

    # Build prompt including all provided modalities
    prompt_text = build_prompt(
        args['prompt'], args['image_path'], args['audio_path'], args['video_path']
    )

    # Prepare generate inputs dictionary
    inputs = {
        'prompt': prompt_text,
        'image_paths': [args['image_path']] if args['image_path'] else [],
        'audio_paths': [args['audio_path']] if args['audio_path'] else [],
        'video_paths': [args['video_path']] if args['video_path'] else [],
        'top_p': args['top_p'],
        'temperature': args['temperature'],
        'max_tgt_len': max_length,
        'stage': args['stage'],
        'freeze_lm': args['freeze_lm'],
        'filter_value': filter_value,
        'min_word_tokens': min_word_tokens,
        'gen_scale_factor': gen_scale_factor,
        'stops_id': stops_id,
        'ENCOUNTERS': ENCOUNTERS,
        'generator': generator,
        # image gen settings
        'load_sd': load_sd,
        'max_num_imgs': max_num_imgs,
        'guidance_scale_for_img': args.get('guidance_scale_for_img'),
        'num_inference_steps_for_img': args.get('num_inference_steps_for_img'),
        # video gen settings
        'load_vd': args.get('load_vd'),
        'max_num_vids': max_num_vids,
        'guidance_scale_for_vid': args.get('guidance_scale_for_vid'),
        'num_inference_steps_for_vid': args.get('num_inference_steps_for_vid'),
        'height': args.get('height'),
        'width': args.get('width'),
        'num_frames': args.get('num_frames'),
        # audio gen settings
        'load_ad': args.get('load_ad'),
        'max_num_auds': args.get('max_num_auds'),
        'guidance_scale_for_aud': args.get('guidance_scale_for_aud'),
        'num_inference_steps_for_aud': args.get('num_inference_steps_for_aud'),
        'audio_length_in_s': args.get('audio_length_in_s'),
    }

    # Generate
    outputs = model.generate(inputs)
    out_dir = os.path.join(os.getcwd(), 'outputs')
    text, media = parse_response(outputs, out_dir)

    print("=== Text Response ===")
    print(text)
    if media:
        print("=== Media saved to ./outputs ===")
        print(media)

if __name__ == '__main__':
    main()
