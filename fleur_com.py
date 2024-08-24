'''
We edit the code from CLIPScore
See in detail https://github.com/jmhessel/clipscore/blob/main/flickr8k_example/compute_metrics.py

Computes the metrics for Flickr8K.
'''

import os
import re
import time

import json, scipy.stats, argparse, torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import TextStreamer

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


def compute_human_correlation(args, tauvariant='c'):
    with open(args.base_fold + '/' + args.input_json) as f:
        total_data = json.load(f)

    # 'PLEASE_CHANGE_IMAGE_FILE_DIR'
    image_folders = {
        "flickr8k" : "flickr8k", # directory containing flickr8k image folder
        "flickr30k" : "flickr30k", # directory containing flickr30k image folder
        "coco" : "coco2014" # directory containing coco2014 image folder
        }

    temperature = 0.2
    num_beams = 1

    device_map="auto"

    kwargs = {"device_map": device_map}
    kwargs['torch_dtype'] = torch.float16
    
    model_path = "liuhaotian/llava-v1.5-13b"
    # model_path = "liuhaotian/llava-v1.5-7b"
    
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path=model_path, model_base=None, model_name=model_name)
    rate2token = {s : tokenizer.encode(str(s))[-1] for s in range(10)}

    conv_mode = "llava_v1"
    
    file_time = time.strftime('%y%m%d%H%M%S', time.localtime(time.time()))
    result_folder = f'./results/'
    os.makedirs(result_folder, exist_ok=True)
    result_file = os.path.join(result_folder, f"fleur_{args.input_json[:-5]}_{file_time}.txt")
    result_file = open(result_file, 'w')
    
    our_scores = []
    human_scores = []

    for data_name, data in total_data.items():
        result_file.write(f'{data_name} dataset\n\n')
        for sample in tqdm(data, desc=data_name):
            image_file = sample["image"]
            candidate = sample["caption"]
            human_scores.append(sample["human"])

            result_file.write(f'image file : {image_file}\n')
            result_file.write(f'caption : {candidate}\n')

            conv = conv_templates[conv_mode].copy()
            roles = conv.roles

            # FLEUR instruction
            inp = f'Your task is to evaluate and rate the caption on a scale of 0.0 to 1.0 based on the given Grading Criteria. (Print Real Number Score ONLY)\n\nGrading Criteria:\n\n0.0: The caption does not describe the image at all.\n1.0: The caption accurately and clearly describes the image.\n\nCaption: {candidate}\n\nScore(Choose a rating from 0.0 to 1.0):'
            outputs = None

            image = Image.open(os.path.join(image_folders[data_name], image_file)).convert('RGB')
            image_tensor = process_images([image], image_processor, args)
            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            
            print(f"{roles[1]}: ", end="")

            if image is not None:
                # first message
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                conv.append_message(conv.roles[0], inp)
                image = None
            else:
                # later messages
                conv.append_message(conv.roles[0], inp)

            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_dict = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample = False, # for deterministic generation
                    temperature=temperature,
                    num_beams=num_beams,
                    max_new_tokens=512,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            output_ids = output_dict.sequences
            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            conv.messages[-1][-1] = outputs
            
            try:
                dotsnumbersdots = re.sub(f'[^\d\.]', '', outputs[:-4])
                numbersdots = re.sub(f'^\.+', '', dotsnumbersdots)
                numbers = re.sub(r'\.+$', '', numbersdots)
                score_check = float(numbers)

                if 0 > score_check or 1 < score_check:
                    continue
                
                if score_check < 1.0:
                    num_index_in_score = str(score_check).index('.') + 1
                    find_num = int(str(score_check)[num_index_in_score])
                    num_index_in_token = (output_ids[0, input_ids.shape[1]:] == rate2token[find_num]).nonzero().squeeze()
                    if len(num_index_in_token.shape) > 0: # if there is a duplication, choose one: e.g., 0.0 -> select the second 0 (after "."), 0.66 -> select the first 6
                        if find_num == 0:
                            num_index_in_token = num_index_in_token[1]
                        else:
                            num_index_in_token = num_index_in_token[0]
                    probs = output_dict.scores[num_index_in_token]
                    probs = torch.nn.functional.softmax(probs, dim=-1)[0]
                    
                    score = 0.
                    for rate, token in rate2token.items(): # score smoothing
                        score += probs[token] * rate * 0.1
                        
                    if len(str(score_check)) > 3: # second decimal place case, 0 < score_check < 1.0
                        num2_index_in_score = str(score_check).index('.') + 2
                        find_num2 = int(str(score_check)[num2_index_in_score])
                        num2_index_in_token = (output_ids[0, input_ids.shape[1]:] == rate2token[find_num2]).nonzero().squeeze()
                        if len(num2_index_in_token.shape) > 0: # if there is a duplication, choose the second one.
                            num2_index_in_token = num2_index_in_token[1]
                        probs2 = output_dict.scores[num2_index_in_token]
                        probs2 = torch.nn.functional.softmax(probs2, dim=-1)[0]
                    
                        for rate, token in rate2token.items():
                            score += probs2[token] * rate * 0.01
                else: # only 1.0 case
                    num_index_in_token = (output_ids[0, input_ids.shape[1]:] == rate2token[1]).nonzero().squeeze()
                    probs = output_dict.scores[num_index_in_token]
                    probs = torch.nn.functional.softmax(probs, dim=-1)[0]
                    score = 0.9 * probs[rate2token[0]] + probs[rate2token[1]]

                result_file.write(f'model score : {score_check}\n')
                result_file.write(f'our score : {score}\n\n')
            except:
                print("Error!")

            our_scores.append(score.cpu())

    result_file.close()

    assert len(our_scores) == len(human_scores)

    print(f'Score Tau-{tauvariant}: {100*scipy.stats.kendalltau(our_scores, human_scores, variant=tauvariant)[0]:.3f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_fold", default='composite', help="annotation file folder")
    parser.add_argument("--input_json", default='composite.json')
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()

    print('COMPOSITE (Tau-c)')
    compute_human_correlation(args, tauvariant='c')
