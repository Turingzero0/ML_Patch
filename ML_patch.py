from ast import literal_eval
import functools
import json
import os
import random
import shutil
import pdb

# Scienfitic packages
import numpy as np
import pandas as pd
import torch
import datasets
from torch import cuda
torch.set_grad_enabled(False)
from tqdm  import tqdm

# Visuals
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(context="notebook",
        rc={"font.size":16,
            "axes.titlesize":16,
            "axes.labelsize":16,
            "xtick.labelsize": 16.0,
            "ytick.labelsize": 16.0,
            "legend.fontsize": 16.0})
palette_ = sns.color_palette("Set1")
palette = palette_[2:5] + palette_[7:]
sns.set_theme(style='whitegrid')

# Utilities

from general_utils import (
  ModelAndTokenizer,
  make_inputs,
  decode_tokens,
  find_token_range,
  predict_from_input,
)

from patchscopes_utils import *

from tqdm import tqdm
tqdm.pandas()
from zhipuai import ZhipuAI
import requests
client = ZhipuAI(api_key="")  
# Here we use ZhipuAI to generate a sentence which contains the subject
# Please use your own api key here.


        





def Ml_patch(model_name, data, patch_num = 3, is_icl=False, only_final_result = False,client = None):
  my_device = torch.device("cuda:4")
  model_to_hook = {
    "EleutherAI/pythia-6.9b": set_hs_patch_hooks_neox,
    "/data3/MODELS/EleutherAI_pythia-12b": set_hs_patch_hooks_neox_batch,
    "meta-llama/Llama-2-13b-hf": set_hs_patch_hooks_llama,
    "lmsys/vicuna-7b-v1.5": set_hs_patch_hooks_llama,
    "./stable-vicuna-13b": set_hs_patch_hooks_llama,
    "CarperAI/stable-vicuna-13b-delta": set_hs_patch_hooks_llama,
    "/data3/MODELS/gpt-j-6b": set_hs_patch_hooks_gptj_batch,
    "/data3/MODELS/Meta-Llama-3-8B-Instruct/":set_hs_patch_hooks_llama,
    "/data3/MODELS/llama2-hf/llama-2-13b-chat":set_hs_patch_hooks_llama_batch,
    "/data3/MODELS/llama2-hf/llama-2-13b":set_hs_patch_hooks_llama_batch,
    "/data3/MODELS/Mistral-7B-Instruct-v0.2":set_hs_patch_hooks_mistral_batch,
    "/data3/MODELS/Qwen/Qwen2.5-7B-Instruct" : set_hs_patch_hooks_qwen_batch,
    "/data/tsq/MODELS/Qwen2.5-1.5B-Instruct" : set_hs_patch_hooks_qwen_batch,
}
  if "13b" in model_name or "12b" in model_name:
    torch_dtype = torch.float16
  else:
    torch_dtype = None
  torch.manual_seed(123)
  np.random.seed(123)
  random.seed(123)
  mt = ModelAndTokenizer(
    model_name,
    low_cpu_mem_usage=False,
    torch_dtype=torch_dtype,
    device=my_device,
  )
  mt.set_hs_patch_hooks = model_to_hook[model_name]
  mt.model.eval()
  df = data
  
  
  

      
  def get_prompt_source(x):
      subject=x['subject']
      while True:
          response = client.chat.completions.create(
          model="glm-4-flash",  # 请填写您要调用的模型名称
          messages=[
              {"role": "user", "content": "Generate exactly one English sentence where the word "+ subject + " appears in the middle (not at the beginning or end). Output only the sentence itself, with no introductory phrases, explanations, or additional text."},
          ],
          )
          position=response.choices[0].message.content.find(subject)
          if (position <10):
              break
      return response.choices[0].message.content
    
    
    
    
  def find_subject_position(x):
      prompt_source_cropped_toks = mt.tokenizer.tokenize(x['prompt_source'])
      text = "The capital city of "+x['subject']
      subject_list = mt.tokenizer.tokenize(text)
      subject = subject_list[-1]
      if x['subject'] in prompt_source_cropped_toks:
          index = prompt_source_cropped_toks.index(x['subject'])
      elif 'Ġ' + x['subject'] in prompt_source_cropped_toks:
          index = prompt_source_cropped_toks.index('Ġ' + x['subject'])
      elif subject in prompt_source_cropped_toks:
          index = prompt_source_cropped_toks.index(subject)
      elif 'Ġ' + subject in prompt_source_cropped_toks:
          subject = 'Ġ' + subject
          index = prompt_source_cropped_toks.index(subject)
      else:
          index = -1

      return int(index)
  
  df.reset_index(drop=True, inplace=True)
  df = df.dropna()
  df["prompt_source"]=df.apply(lambda x: get_prompt_source(x),axis=1)
  df["prompt_target"] = "The " + df['relation'] + " x"
  df['position_source']=df.apply(lambda x: find_subject_position(x),axis=1)
  df = df[df['position_source'] > 0].reset_index(drop=True)
  df = df[df['position_source'] < 10].reset_index(drop=True)
  df['position_target'] = int(-1)
  batch_size = 256
  # Dropping empty prompt sources. This is an artifact of saving and reloading inputs
  df = df[~df["prompt_source"].apply(lambda x: isinstance(x, float))].reset_index(drop=True)
  # Dropping prompt sources with \n. pandas read_pickle is not able to handle them properly and drops the rest of the input.
  df = df[~df["prompt_source"].str.contains('\n')].reset_index(drop=True)
      # BATCHED
  batch = []
  for _, row in tqdm(df.iterrows()):
      for layer_source in range(mt.num_layers - patch_num + 1):
          for layer_target in range(mt.num_layers - patch_num  + 1):
              item = dict(row)
              item.update({
                  "layer_source": layer_source,
                  "layer_target": layer_target,
              })
              batch.append(item)
  experiment_df = pd.DataFrame.from_records(batch)
  eval_results = evaluate_attriburte_exraction_batch(mt, experiment_df, batch_size=batch_size, is_icl=is_icl, patch_num = patch_num)
  results_df = experiment_df.head(len(eval_results["is_correct_patched"]))
  for key, value in eval_results.items():
    results_df[key] = list(value)

  # evaluate
  target_layer = mt.num_layers - patch_num
  df = results_df[results_df['layer_source']<=target_layer].reset_index(drop=True)
  df1 = pd.DataFrame(columns=['subject','prompt_source','prompt_target','layer_source','is_correct_patched','generations'])
  for index, row in df.iterrows():
      if df1.empty:
          df1.loc[len(df1)] = [row['subject'],row['prompt_source'],row['prompt_target'], row['layer_source'], row['is_correct_patched'],row['generations_patched_postprocessed']]
          continue
      if row['prompt_source'] != df1.iloc[-1]['prompt_source']:
          df1.loc[len(df1)] = [row['subject'],row['prompt_source'],row['prompt_target'], row['layer_source'], row['is_correct_patched'],row['generations_patched_postprocessed']]
      else:
          if row['layer_source'] > df1.iloc[-1]['layer_source']:
              df1.loc[len(df1)] = [row['subject'],row['prompt_source'],row['prompt_target'], row['layer_source'], row['is_correct_patched'],row['generations_patched_postprocessed']]
          if row['is_correct_patched'] == True:
              df1.iloc[-1]['is_correct_patched'] = True
              df1.iloc[-1]['generations'] = row['generations_patched_postprocessed']
  if not only_final_result:
    return df1  
  else:
    # 对于同一个subject和object，如果有一行的is_correct_patched为True，则返回第一个is_correct_patched为True的行
    # 如果没有is_correct_patched为True的行，则返回第一个generations不为空的行
    # 创建一个结果列表
    filtered_rows = []

    # 按 subject 和 prompt_target 分组
    grouped = df1.groupby(['subject', 'prompt_target'])

    for (subject, prompt_target), group in grouped:
        # 检查是否有 is_correct_patched 为 True 的行
        correct_patched_rows = group[group['is_correct_patched'] == True]
        if not correct_patched_rows.empty:
            # 如果存在，选择第一个 is_correct_patched 为 True 的行
            filtered_rows.append(correct_patched_rows.iloc[0])
        else:
            # 如果不存在，选择第一个 generations 不为空的行
            non_empty_generations_rows = group[group['generations'].notna() & (group['generations'] != "")]
            if not non_empty_generations_rows.empty:
                filtered_rows.append(non_empty_generations_rows.iloc[0])
    
    df1 = pd.DataFrame(filtered_rows)
    return df1
