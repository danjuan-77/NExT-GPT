import argparse
import os
import tempfile
import json
import torch
import scipy.io.wavfile
import imageio
from PIL import Image
from model.anyToImageVideoAudio import NextGPTModel
from config import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Command-line multimodal inference (text, image, audio, video) for NExT-GPT"
    )
    # Input modalities
    parser.add_argument("--text", type=str, default=None, help="Input text prompt")
    parser.add_argument("--image", type=str, default=None, help="Path to input image file")
    parser.add_argument("--audio", type=str, default=None, help="Path to input audio file (wav)")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file")
    # Optional flags
    parser.add_argument(
        "--use_video_audio",
        action="store_true",
        help="Include video’s embedded audio during inference"
    )
    # Model & checkpoint settings
    parser.add_argument(
        "--model", type=str, default="nextgpt", help="Model name identifier"
    )
    parser.add_argument(
        "--nextgpt_ckpt_path",
        required=True,
        help="Path to checkpoint directory containing pytorch_model.pt"
    )
    parser.add_argument(
        "--stage", type=int, default=3, help="Inference stage identifier"
    )
    parser.add_argument(
        "--freeze_lm",
        type=bool,
        default=True,
        help="Freeze language model parameters during inference"
    )
    # Generation hyperparameters
    parser.add_argument("--top_p", type=float, default=0.01, help="Top-p sampling parameter")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--max_length", type=int, default=246, help="Maximum generated token length")
    return parser.parse_args()


def load_model(args):
    # Load base config and merge CLI args
    cfg = load_config(vars(args))
    # Ensure 'stage' is provided for NextGPTModel
    cfg['stage'] = args.stage

    # Initialize model
    model = NextGPTModel(**cfg)
    # Load delta checkpoint
    ckpt_file = os.path.join(args.nextgpt_ckpt_path, "pytorch_model.pt")
    state = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(state, strict=False)

    # Optionally freeze language model weights
    if args.freeze_lm:
        for param in model.language_model.parameters():
            param.requires_grad = False

    # Move to eval GPU
    model = model.eval().half().cuda()
    return model


def save_image(img):
    os.makedirs("temp", exist_ok=True)
    path = os.path.join("temp", f"{next(tempfile._get_candidate_names())}.jpg")
    img.save(path)
    return path


def save_video(frames):
    os.makedirs("temp", exist_ok=True)
    path = os.path.join("temp", f"{next(tempfile._get_candidate_names())}.mp4")
    writer = imageio.get_writer(path, fps=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    return path


def save_audio(data, rate=16000):
    os.makedirs("temp", exist_ok=True)
    path = os.path.join("temp", f"{next(tempfile._get_candidate_names())}.wav")
    scipy.io.wavfile.write(path, rate, data)
    return path


def parse_response(outputs):
    """
    Parse model outputs: list of dicts with keys 'text', 'img', 'vid', 'aud'.
    Returns JSON-serializable dict with text and file paths.
    """
    result = {"text": [], "images": [], "videos": [], "audios": []}
    for item in outputs:
        if isinstance(item, str):
            result["text"].append(item)
        elif 'img' in item:
            for img in item['img']:
                if hasattr(img, 'save'):
                    result["images"].append(save_image(img[0]))
        elif 'vid' in item:
            for frames in item['vid']:
                result["videos"].append(save_video(frames))
        elif 'aud' in item:
            for aud in item['aud']:
                result["audios"].append(save_audio(aud))
    # Clean text
    result["text"] = [t.strip() for t in result["text"] if t.strip()]
    return result


def main():
    args = parse_args()
    model = load_model(args)

    # Build generation inputs
    inputs = {
        'prompt': args.text or "",
        'image_paths': [args.image] if args.image else [],
        'audio_paths': [args.audio] if args.audio else [],
        'video_paths': [args.video] if args.video else [],
        'top_p': args.top_p,
        'temperature': args.temperature,
        'max_tgt_len': args.max_length,
        'modality_embeds': None,
        'filter_value': -float('Inf'),
        'min_word_tokens': 10,
        'gen_scale_factor': 4.0,
        'max_num_imgs': 1,
        'max_num_vids': 1,
        'max_num_auds': 1,
        'stops_id': [[835]],
        'load_sd': True,
        'generator': torch.Generator(device='cuda').manual_seed(13),
    }

    # Handle video vs video+audio
    if args.video and args.use_video_audio:
        inputs['audio_paths'] += [args.video]

    # Perform generation
    raw_outputs = model.generate(inputs)
    parsed = parse_response(raw_outputs)

    # Print results
    print("\n=== Generated Text ===")
    for line in parsed['text']:
        print(line)
    if parsed['images']:
        print("\nSaved Images:")
        for p in parsed['images']:
            print(f" - {p}")
    if parsed['videos']:
        print("\nSaved Videos:")
        for p in parsed['videos']:
            print(f" - {p}")
    if parsed['audios']:
        print("\nSaved Audios:")
        for p in parsed['audios']:
            print(f" - {p}")

    # JSON output
    print("\n=== JSON Output ===")
    print(json.dumps(parsed, indent=2))


if __name__ == '__main__':
    main()
