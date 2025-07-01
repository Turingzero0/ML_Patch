# ML-Patch: Carefully Evaluating Hidden Knowledge of Language Models via Multi-layer Patching

We present **ML-Patch**, a new evaluation method of LLMs, which consists of an [online website](http://103.235.229.133:9622/) to show our method clealy and an easy-use [online inference website](https://huggingface.co/spaces/lllyx/ML_Patch) which can infer the result of ML-Patch online using user's own data. Moreover, we also provide an offline python [toolkit](https://github.com/Turingzero0/ML_Patch/blob/master/api.ipynb) for users who want to upload large amounts of data.

Specifically, we propose a new method to evaluate the knowledge boundry pf LLMs, which can make better use of the **hidden states** of LLMs. It is significantly different from today's evaluation methods which most base on prompt.

We test our method on a series of pretrained models, including **llama2-13b** ,**gpt-j-6b**, **Qwen2.5-7b** e.t.c. The results show that our method can effectively evaluate the knowledge of LLMs.

![image](patch.jpg)

## Download data

We use the factual triples sorted out from wikidata.

[Data address](https://github.com/Turingzero0/ML_Patch/blob/master/factual_pkl)

## Quick start

By running [api.ipynb](https://github.com/Turingzero0/ML_Patch/blob/master/api.ipynb), you can input the factual knowledge and choose a series of hyperparameters such as `model` and get a pkl and tsv file which contain the final results.

```
import pandas as pd
import io
import os
from ML_patch import *
from zhipuai import ZhipuAI
import requests
client = ZhipuAI(api_key="")  
# Here we use ZhipuAI to generate a sentence which contains the subject
# Please use your own api key here.

data_ = "id,subject,relation,object\n001,France,capital city of,Paris"
bytes_io = io.StringIO(data_)
df = pd.read_csv(bytes_io, sep=",") # You can load your own data here
result = Ml_patch(model_name= "/data3/MODELS/gpt-j-6b" , data = df, only_final_result= False, patch_num= 3,client=client)
result.to_csv("./patch.tsv", sep="\t", index=False)
result.to_pkl("./patch.pkl")
```

More hyperparameters can be adjusted in [ML_Patch.py](https://github.com/Turingzero0/ML_Patch/blob/master/ML_patch.py)
