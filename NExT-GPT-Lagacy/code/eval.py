import os
import argparse
import tempfile
import torch
import imageio
import scipy.io
from PIL import Image
from model.anyToImageVideoAudio import NextGPTModel
from config import load_config
import tempfile
import traceback
from tqdm import tqdm
import json
from typing import List, Optional
from moviepy.editor import (
    AudioFileClip,
    concatenate_audioclips,
    ImageClip,
    concatenate_videoclips,
)
tempfile.tempdir = "/share/nlp/tuwenming/projects/HAVIB/tmp"
maic_cls_list = ['bus', 'hair-dryer', 'pipa', 'man', 'ambulance', 'razor', 'harp', 'tabla', 'bass', 'handpan', 
        'girl', 'sitar', 'car', 'lion', 'guitar', 'vacuum-cleaner', 'cat', 'mower', 'helicopter', 'boy', 'drum', 
        'keyboard', 'tuba', 'saw', 'flute', 'cello', 'woman', 'gun', 'accordion', 'violin', 'clarinet', 'erhu', 
        'saxophone', 'guzheng', 'dog', 'baby', 'horse', 'male', 'wolf', 'bird', 'ukulele', 'piano', 'female', 
        'marimba', 'not sure', 'no available option']

mvic_cls_list = ['sushi', 'banana', 'cake', 'butterfly', 'bird', 'microphone', 'hamburger', 'pineapple', 
        'man', 'book', 'sunglasses', 'goat', 'tie', 'cabinetry', 'motorcycle', 'drawer', 'strawberry', 
        'sheep', 'pasta', 'parrot', 'bull', 'table', 'penguin', 'watch', 'pillow', 'shellfish', 'kangaroo', 
        'flower', 'paddle', 'rocket', 'helicopter', 'bus', 'mushroom', 'bee', 'tree', 'boat', 'saxophone', 
        'football', 'lizard', 'violin', 'dog', 'cucumber', 'cello', 'airplane', 'horse', 'drum', 'box', 
        'rabbit', 'car', 'door', 'orange', 'shelf', 'camera', 'poster', 'lemon', 'cat', 'fish', 'bread', 
        'piano', 'apple', 'glasses', 'bicycle', 'truck', 'deer', 'woman', 'wheelchair', 'cheese', 'chair', 
        'plate', 'tomato', 'bed', 'starfish', 'balloon', 'bottle', 'crab', 'beer', 'frog', 'shrimp', 'tower', 
        'guitar', 'pig', 'peach', 'train', 'pumpkin', 'elephant', 'jellyfish', 'parachute', 'monkey', 'flag',
        'not sure', 'no available option']

prompt_avl = """
        In each video frame, there may be multiple categories of sound-emitting instances. Each category can have several instances. 
        You can choose instance categories from the given categories list.
        The naming format for instances is: category_id. Instance IDs start from 1, e.g., male_1, dog_2, dog_3, cat_4. 
        It is crucial that the instance names (i.e., category_id) remain consistent for the same instances across different frames.
        The bbox format is: [x, y, w, h], where x and y represent the coordinates of the top-left corner, and w and h are the width and height. 
        The final answer must strictly adhere to the following format: 
        answer={"frame_0": {"guzheng_1": "[269, 198, 83, 16]", "guzheng_2": "[147, 196, 75, 13]", "female_3": "[152, 108, 123, 36]"}, "frame_1": ..., "frame_n": ...}
    """

avl_cls_list = ['dog', 'clarinet', 'banjo', 'cat', 'guzheng', 'tree', 'lion', 'tuba', 
        'ukulele', 'flute', 'piano', 'person', 'violin', 'airplane', 'bass', 'pipa', 
        'trumpet', 'accordion', 'saxophone', 'car', 'lawn-mower', 'cello', 'bassoon', 
        'horse', 'guitar', 'erhu', 'not sure', 'no available option']

prompt_avlg = """
        Please output the answer in a format that exactly matches the following example:
        answer={'frame_0': [x0, y0, w0, h0], 'frame_1': None, ..., 'frame_9': [x9, y9, w9, h9]}.
        Note, for [x, y, w, h], where x and y represent the top-left corner of the bounding box, 
        and w and h represent the width and height of the bounding box.
    """

avqa_cls_list = ['ukulele', 'cello', 'clarinet', 'violin', 'bassoon', 'accordion', 'banjo', 'tuba', 'flute', 'electric_bass', 'bagpipe', 
        'drum', 'congas', 'suona', 'xylophone', 'saxophone', 'guzheng', 'trumpet', 'erhu', 'piano', 'acoustic_guitar', 'pipa', 'not sure', 'no available option']

havib_constants = {
    'L1_LAQA': {
        'options_sound_clarity': ['first', 'last', 'same', 'not sure'],
        'options_sound_order': ['sound', 'noise', 'not sure'],
        'options_sound_volume': ['first', 'last', 'same', 'not sure'],
        'options_yes_no': ['yes', 'no', 'not sure'],
    },

    'L1_LIQA': {
        'get_from_background_binary': ['yes', 'no', 'not sure'],
        'get_from_image_binary': ['yes', 'no', 'not sure'],
        'get_from_foreground_binary': ['yes', 'no', 'not sure'],
        'get_from_image_triple': ['blurred', 'normal', 'clear', 'not sure'],
        'get_from_3d-task1': ['center', 'left', 'right', 'not sure'],
        'get_from_3d-task2': ['cone', 'cube', 'cylinder', 'cuboid', 'no available option', 'not sure'],
        # 'get_from_3d-task3': [0, 1, 2, 3, 4, 5, 6],
        'get_from_space_hard': ['center', 'top left', 'top center', 'top right', 'bottom left', 'bottom center', 'bottom right', 'no available option', 'not sure'],
        'get_from_color': ['blue', 'green', 'red', 'puprle', 'yellow', 'no available option', 'not sure'],
        'get_yes_no': ['yes', 'no', 'not sure'],
        # 'get_lines_count': [0, 1, 2, 3, 4],
        'get_lines_direction': ['horizontal', 'vertical', 'inclined', 'not sure'],
        'get_from_space_easy_area': ['the right one', 'the left one', 'the middle one', 'the bottom one', 'the top one'],
        'get_from_space_easy_bbrightness': ['the right one', 'the left one', 'the middle one', 'the bottom one', 'the top one'],
        'options_yes_no': ['yes', 'no', 'not sure'],
    },

    'L1_LVQA': {
        'which_object': ['square', 'circle', 'triangle', 'not sure', 'no available option', 'not sure'],
        'what_shape': ['Triangular pyramid', 'Cone', 'Cube', 'Sphere', 'None', 'not sure'],
        # 'how_many': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'what_movement_2d': ['horizontal', 'inclined', 'vertical', 'no movenment', 'None', 'not sure'],
        'what_movement_3d': ['Rotation', 'Shrinking', 'Translation', 'Enlarging', 'None', 'not sure'],
        'what_surface': ['Rough', 'Moderate', 'Smooth', 'None', 'not sure'],
        'spacial_change': ['Bottom-left to top-right', 'Bottom-right to top-left', 'Top-left to bottom-right', 'Top-right to bottom-left', 'None', 'not sure', 'No movement',],
        'options_yes_no': ['yes', 'no', 'not sure'],
    },

    'L2_MAIC': {
        'maic_cls_list': maic_cls_list,
        'prompt_maic': "There may be one or more sound-emitting objects in the provided audio. \nPlease strictly output the answer in the format answer={'cls_1': count_1, 'cls_2': count_2}, \nfor example, answer={'dog': 2, 'cat': 3, 'male': 1}. \n"
    },

    'L2_MVIC': {
        'mvic_cls_list': mvic_cls_list,
        'prompt_mvic': "There may be one or more visible objects in the provided image. \nPlease strictly output the answer in the format answer={'cls_1': count_1, 'cls_2': count_2}, \nfor example, answer={'dog': 2, 'cat': 3, 'male': 1}. \n Possible categoris are in the list: mvic_cls_list"
    },

    'L3_AVH': {
        'prompt_avh': "Please answer the question based on the given audio and video.",
        'avh_options_list': ['yes', 'no', 'not sure'],
    },

    'L3_VAH': {
        'prompt_vah': "Please answer the question based on the given audio and video.",
        'vah_options_list': ['yes', 'no', 'not sure'],
    },

    'L3_AVL': {
        'prompt_avl': prompt_avl,
        'avl_cls_list': avl_cls_list,
    },

    'L3_AVM': {
        'prompt_avm': 'Please answer the question based on the given audio and video.',
        'avm_options_list': ['yes', 'no', 'not sure'],
    },

    'L3_AVR': {
        'prompt_avr': "Please output the indices of the images list, starting from 0. For example: [], or [0, 3], or [1, 4, 9]."
    },

    'L3_VAR': {
        'prompt_var': "Please output the indices of the wavs list, starting from 0. For example: [], or [0, 3], or [1, 4, 9]."
    },

    'L4_AVC': {

    },

    'L4_AVLG': {
        'prompt_avlg': prompt_avlg,
    },

    'L4_AVQA': {
        'avqa_options_list_is': ['yes', 'no', 'not sure'],
    },

    'L5_AVLG': {
        'prompt_avlg': prompt_avlg,
    },

    'L5_AVQA': {
        'avqa_cls_list': avqa_cls_list,
        'avqa_options_list_is': ['yes', 'no', 'not sure'],
    },
}
def concat_audio(audio_paths: List[str]) -> str:
    """
    Concatenate multiple audio files into one WAV file.
    Returns the path to the temp WAV file.
    """
    clips = [AudioFileClip(p) for p in audio_paths]
    final = concatenate_audioclips(clips)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_path = tmp.name
    final.write_audiofile(out_path, fps=16000, logger=None)
    return out_path

def images_to_video(image_paths: List[str], duration: float, fps: int = 1) -> str:
    """
    Turn a list of images into a silent video of total `duration` seconds.
    Each image is shown for `duration / len(image_paths)` seconds.
    Returns the path to the temp MP4 file.
    """
    single_dur = duration / len(image_paths)
    clips = [ImageClip(p).set_duration(single_dur) for p in image_paths]
    video = concatenate_videoclips(clips, method="compose")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp.name
    video.write_videofile(out_path, fps=fps, codec="libx264", audio=False, logger=None)
    return out_path

def images_and_audio_to_video(image_paths: List[str], audio_paths: List[str], fps: int = 1) -> str:
    """
    Concatenate audio_paths into one audio, then build a video from image_paths
    that matches the audio duration, and merge them.
    Returns the path to the temp MP4 file.
    """
    # 1) build the concatenated audio
    audio_path = concat_audio(audio_paths)
    audio_clip = AudioFileClip(audio_path)
    # 2) build video from images matching audio duration
    duration = audio_clip.duration
    vid_path = images_to_video(image_paths, duration, fps=fps)
    # 3) attach audio to video
    video_clip = AudioFileClip(audio_path)  # re-open to avoid MoviePy caching issues
    from moviepy.editor import VideoFileClip
    base_vid = VideoFileClip(vid_path)
    final = base_vid.set_audio(audio_clip)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp.name
    final.write_videofile(out_path, fps=fps, codec="libx264", logger=None)
    return out_path 
    
def get_real_path(task_path: str, src_path: str) -> str:
    """传入taskpath和一些文件的path，构造文件的真实path

    Args:
        task_path (str): task path
        src_path (str): 每个文件的path

    Returns:
        str: 文件的真实path
    """
    temp_path = os.path.join(task_path, src_path)
    return os.path.normpath(temp_path)

def get_real_options_or_classes(d: dict) -> str:
    """Replace pseudo-options with real options text."""
    opts = d['input']['question'].get('options')
    if opts in havib_constants.get(d['task'], {}):
        opts = havib_constants[d['task']][opts]
    if opts:
        label = 'semantic categories' if 'cls' in opts else 'options'
        return f"Available {label} are: {opts}"
    return ''

def get_real_prompt(d: dict) -> str:
    """Replace pseudo-prompt with real prompt text."""
    prm = d['input']['question'].get('prompt')
    if prm in havib_constants.get(d['task'], {}):
        prm = havib_constants[d['task']][prm]
    return prm or ''

def get_real_input(d: dict) -> str:
    """Concatenate prompt, options, and question text into one input string."""
    prompt = get_real_prompt(d)
    options = get_real_options_or_classes(d)
    question = d['input']['question']['text'] or ''
    # 去掉多余的句点
    parts = [p for p in (prompt, options, question) if p]
    return " ".join(parts)
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


def build_prompt(prompt: str, image_path: str=None, audio_path: str=None, video_path: str=None, history=None) -> str:
    """
    Build the full prompt string with modality tags and optional chat history.
    """
    history = history or []
    text = ''
    if not history:
        text += '### Human: '
    else:
        for i, (q, a) in enumerate(history):
            sep = '###' if i == 0 else ''
            text += f'{sep} Human: {q}\n### Assistant: {a}\n'
        text += '### Human: '
    if image_path:
        text += f'<Image>{image_path}</Image> '
    if audio_path:
        text += f'<Audio>{audio_path}</Audio> '
    if video_path:
        text += f'<Video>{video_path}</Video> '
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
    # parser.add_argument('--prompt', type=str, default='', help='Text prompt')
    # parser.add_argument('--image_path', type=str, default=None, help='Path to an input image')
    # parser.add_argument('--audio_path', type=str, default=None, help='Path to an input audio file (wav)')
    # parser.add_argument('--video_path', type=str, default=None, help='Path to an input video file')

    # Sampling options
    parser.add_argument('--top_p', type=float, default=0.01)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max_tgt_len', type=int, default=246)
    
    parser.add_argument(
        "--task_path",
        type=str,
        required=True,
        help="Path to the task folder containing data.json and media files"
    )

    args = vars(parser.parse_args())
    args.update(load_config(args))
    
    # Initialize and load model
    model = NextGPTModel(**args)
    ckpt = torch.load(os.path.join(args['nextgpt_ckpt_path'], 'pytorch_model.pt'), map_location='cuda')
    model.load_state_dict(ckpt, strict=False)
    model.eval().half().cuda()
    
    
    task_path = args['task_path']
    task_name = f"L{task_path.rsplit('/', 1)[0][-1]}_{task_path.rsplit('/', 1)[-1]}"
    model_name = "nextgpt"
    save_prediction_json = f'/share/nlp/tuwenming/projects/HAVIB/eval/user_outputs/{model_name}/tasks/{task_name}.json'
    os.makedirs(os.path.dirname(save_prediction_json), exist_ok=True)
    print('>>> save res to:', save_prediction_json)


    data_json_path = os.path.join(task_path, "data.json")
    with open(data_json_path, "r", encoding='utf-8') as f:
        raw_data = json.load(f)
    print(">>>Finished load raw data...")
    parsed_data = []
    for item in raw_data:
        inp = item.get('input', {})
        question = inp.get('question', {})
        entry = {
            'id': item.get('id'),
            'task': item.get('task'),
            'subtask': item.get('subtask', None),
            'text': get_real_input(item),
            'audio_list': inp.get('audio_list', None),
            'image_list': inp.get('image_list', None),
            'video': inp.get('video', None)
        }
        parsed_data.append(entry)

    print(">>>Finished parse raw data...")

    predictions = []
    
    for data in tqdm(parsed_data):
        _id = data['id']
        _task = data['task']
        _subtask = data['subtask']
        text = data['text']
        audio_list = (
            [get_real_path(task_path, p) for p in data["audio_list"]]
            if data["audio_list"] else None
        )
        image_list = (
            [get_real_path(task_path, p) for p in data["image_list"]]
            if data["image_list"] else None
        )
        video = (
            get_real_path(task_path, data['video'])
            if data['video'] else None
        )
        print(f">>> text input=:{text}")
        
        # try:
        # Case 1: only audio_list
        if audio_list and not image_list and not video:
            if len(audio_list) > 1:
                audio_path = concat_audio(audio_list)
            else:
                audio_path = audio_list[0]
            prompt_text =  build_prompt(text, audio_path=audio_path)

        # Case 2: only one image
        elif image_list and not audio_list and not video:
            image_path = image_list[0]
            prompt_text =  build_prompt(text, image_path=image_path)

        # Case 3: only video
        elif video and not audio_list and not image_list:
            prompt_text = build_prompt(text, video_path=video)

        # Case 4: video + audio_list
        elif video and audio_list:
            prompt_text = build_prompt(text, audio_path=audio_list[0], video_path=video)

        # Case 5: image_list + audio_list
        elif image_list and audio_list and not video:
            video_path = images_and_audio_to_video(image_list, audio_list, fps=1)
            prompt_text = build_prompt(text, audio_path=video_path, video_path=video_path)

        # Case 6: audio_list + video (same as Case 4)
        elif audio_list and video:
            prompt_text = build_prompt(text, audio_path=audio_list[0], video_path=video)    

        # Prepare generate inputs dictionary
        inputs = {
            'prompt': prompt_text,
            'image_paths': image_list if image_list else [],
            'audio_paths': audio_list if audio_list else [],
            'video_paths': video if video else [],
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
        predict_text, media = parse_response(outputs, out_dir)
        # except Exception as e:
        #     # 捕获任何异常，并把完整 traceback 当作 output
        #     tb = traceback.format_exc()
        #     predict_text = f"Error during inference:\n{tb}"
        
        pred_record = {
            "task": _task,
            "subtask": _subtask,
            "id": _id,
            "predict": predict_text,
        }
        predictions.append(pred_record)
        print('>>> ans=:', pred_record)
    
    
    with open(save_prediction_json, 'w', encoding='utf-8') as json_file:
        json.dump(predictions, json_file, ensure_ascii=False, indent=4)
    
    

if __name__ == '__main__':
    main()
