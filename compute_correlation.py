import numpy as np
import os
import json
import scipy.stats

base_fold = 'flickr8k'
ann_file = 'flickr8k.json'
# ann_file = 'crowdflower_flickr8k.json' # tau-b

data = {}
with open(os.path.join(base_fold, ann_file)) as f:
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

scores_raw = []
scores_ss = []
path = './results/PLEASE_CHANGE_FILE_NAME.txt'

with open(path, 'r') as f:
    for line in f:
        if line.startswith('model'):
            scores_raw.extend([float(line.split()[-1])] * 3) # for Flickr8k-Expert
            # scores_model.append(float(line.split()[-1]))
        if line.startswith('our'):
            scores_ss.extend([float(line.split()[-1])] * 3) # for Flickr8k-Expert
            # scores_ss.append(float(line.split()[-1]))

print(f'Raw score Tau-c: {100*scipy.stats.kendalltau(scores_raw, human_scores, variant="c")[0]:.3f}')
print(f'FLEUR Tau-c: {100*scipy.stats.kendalltau(scores_ss, human_scores, variant="c")[0]:.3f}')