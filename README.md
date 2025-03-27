# ML-Patch: Carefully Evaluating Hidden Knowledge of Language Models via Multi-layer Patching

We present **ML-Patch**, a new evaluation method of LLMs, which consists of an [online website](http://118.26.161.195:9622/#/) to show our method clealy and a easy-use [toolkit](https://github.com/Turingzero0/ML_Patch/blob/master/api.ipynb).

Specifically, we propose a new method to evaluate the knowledge boundry pf LLMs, which can make better use of the **hidden states** of LLMs. It is significantly different from today's evaluation methods which most base on prompt.

We test our method on a series of pretrained models, including **llama2-13b** ,**gpt-j-6b**, **Qwen2.5-7b** e.t.c. The results show that our method can effectively evaluate the knowledge of LLMs.

![image](patch.jpg)

## Download data

We use the factual triples sorted out from wikidata.

[Data address](https://github.com/PAIR-code/interpretability/tree/master/patchscopes/code/preprocessed_data/factual)

## Quick start

By running [api.ipynb](https://github.com/Turingzero0/ML_Patch/blob/master/api.ipynb), you can input the factual knowledge and choose a series of hyperparameters such as `model` and get a pkl and tsv file which contain the final results.

```
import pandas as pd
import io
import os
from ML_patch import *

data_ = "id,subject,relation,object\n001,France,capital city of,Paris"
bytes_io = io.StringIO(data_)
df = pd.read_csv(bytes_io, sep=",")
result = Ml_patch(model_name= "/data3/MODELS/gpt-j-6b" , data = df, only_final_result= False)
result.to_csv("./patch.tsv", sep="\t", index=False)
result.to_pkl("./patch.pkl")
```

More hyperparameters can be adjusted in [ML_Patch.py](https://github.com/Turingzero0/ML_Patch/blob/master/ML_patch.py)
