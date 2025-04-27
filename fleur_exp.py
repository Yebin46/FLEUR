import argparse
import torch

import LLaVA.llava as llava
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
from transformers import TextStreamer

import os
import re
import time
import json
import numpy as np
import scipy.stats
from tqdm import tqdm


def main(args):
    # Model
    disable_torch_init()

    base_fold = './flickr8k'
    ann_file = 'flickr8k.json'

    data = {}
    with open(base_fold + '/' + ann_file) as f:
        data.update(json.load(f))

    img_list = []
    txt_list = []
    human_scores = []
    for k, v in list(data.items()):
        for human_judgement in v['human_judgement']:
            if np.isnan(human_judgement['rating']):
                print('NaN')
                continue
            human_scores.append(human_judgement['rating'])
            img_list.append(human_judgement['image_path'])
            txt_list.append(' '.join(human_judgement['caption'].split()))
            
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    rate2token = {s : tokenizer.encode(str(s))[-1] for s in range(10)}

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode


    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles
    
    file_time = time.strftime('%y%m%d%H%M%S', time.localtime(time.time()))
    result_folder = f'./results/'
    result_file = os.path.join(result_folder, f"fleur_exp_{ann_file[:-5]}_{file_time}.txt")
    result_file = open(result_file, 'w')

    our_scores = []
    for idx, (image_file, candidate) in enumerate(zip(tqdm(img_list), txt_list)):
        if idx % 3 == 0:
            result_file.write(f'image file : {image_file}\n')
            result_file.write(f'caption : {candidate}\n')
            
            conv = conv_templates[conv_mode].copy()
            roles = conv.roles
            
            image = Image.open(os.path.join('PLEASE_CHANGE_IMAGE_FILE_DIR', image_file)).convert('RGB')
            image_tensor = process_images([image], image_processor, args)
            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)

            inputs = [f"Your task is to evaluate and rate the caption on a scale of 0.0 to 1.0 based on the given Grading Criteria. (Print Real Number Score ONLY)\n\nGrading Criteria:\n\n0.0: The caption does not describe the image at all.\n1.0: The caption accurately and clearly describes the image.\n\nCaption: {candidate}\n\nScore(Choose a rating from 0.0 to 1.0):",
                    "Why? Tell me the reason."]

            flag = 1
            for inp in inputs:

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
                    print(f"{roles[1]}: ", end="")
                    output_dict = model.generate(
                                input_ids,
                                images=image_tensor,
                                do_sample = False,
                                temperature=args.temperature,
                                num_beams=1,
                                max_new_tokens=512,
                                streamer=streamer if flag == 0 else None,
                                use_cache=True,
                                stopping_criteria=[stopping_criteria],
                                output_scores=True,
                                return_dict_in_generate=True,
                            )
                output_ids = output_dict.sequences
                outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

                if flag:
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
                        if len(num_index_in_token.shape) > 0:
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
                            if len(num2_index_in_token.shape) > 0:
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
                    
                    conv.messages[-1][-1] = str(score.item()) + '</s>'
                    print(score.item())
                    print("\nRaw Score: ", outputs[:-4], "\n")

                    result_file.write(f'model score : {score_check}\n')
                    result_file.write(f'our score : {score}\n')
                    
                    flag = 0
                    our_scores.extend([score.cpu()]*3)
                else:
                    conv.messages[-1][-1] = outputs
                    result_file.write(f'explanation : {outputs}\n\n')
        else:
            continue

    result_file.close()
    print(f"Score Tau-c: {100*scipy.stats.kendalltau(our_scores, human_scores, variant='c')[0]:.3f}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()
    main(args)
