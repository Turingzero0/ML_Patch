# ML-Patch: Carefully Evaluating Hidden Knowledge of Language Models via Multi-layer Patching

We present **ML-Patch**, a new evaluation method of LLMs, which consists of an [online website](http://118.26.161.195:9622/#/) to show our method clealy and a easy-use [toolkit](https://github.com/Turingzero0/ML_Patch/blob/master/api.ipynb).

Specifically, we propose a new method to evaluate the knowledge boundry pf LLMs, which can make better use of the hidden states of LLMs. It is significantly different from today's evaluation methods which most base on prompt.

Our method can be even used on **pretrained models** which have a wider range of applications.

![image](patch.jpg)

## Download data

We use the factual triples sorted out from wikidata.

[Data address](https://github.com/PAIR-code/interpretability/tree/master/patchscopes/code/preprocessed_data/factual)

## Quick start

By running [api.ipynb](https://github.com/Turingzero0/ML_Patch/blob/master/api.ipynb), you can input the factual knowledge and choose a series of hyperparameter such as `model` and get a pkl and tsv file which contain the final results.
