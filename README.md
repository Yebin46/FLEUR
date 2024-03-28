# FLEUR
FLEUR: An Explainable Reference-Free Evaluation Metric for Image Captioning Using a Large Multimodal Model


## Getting Started
FLEUR utilizes the LLaVA model for performing image caption evaluation (though you may use other Vision Language Models if desired). Please follow the instructions in the [LLaVA GitHub README](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#install) for the necessary setup. **No additional training is required.**


## Evaluation on Flickr8k-Expert dataset
* Running code for FLEUR:
```
CUDA_VISIBLE_DEVICES=0,1 python fleur.py
```
* Running code for RefFLEUR:
```
CUDA_VISIBLE_DEVICES=0,1 python reffleur.py
```
## Or get the explanation together
```
CUDA_VISIBLE_DEVICES=0,1 python fleur_exp.py
```

The evaluation result will be saved as txt files in the `results` folder.

## Compute Kendall's Tau Correlation
Change file names of annotation file and the evaluation result file in `compute_correlation.py`
```
python compute_correlation.py
```