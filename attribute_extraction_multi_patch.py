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
}
# Load model

# 0-shot with GPT-J 
model_name = "/data3/MODELS/EleutherAI_pythia-12b"
sos_tok = False

if "13b" in model_name or "12b" in model_name:
    torch_dtype = torch.float16
else:
    torch_dtype = None

my_device = torch.device("cuda:1")

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

def run_experiment(task_type, task_name, data_dir, output_dir, batch_size=512, n_samples=-1,
                   save_output=True, replace=False, only_correct=False, is_icl=True):
    fdir_out = f"{output_dir}/{task_type}"
    fname_out = f"{fdir_out}/{task_name}_only_correct_{only_correct}.pkl"
    # if not replace and os.path.exists(fname_out):
    #     print(f"File {fname_out} exists. Skipping...")
    #     return
    print(f"Running experiment on {task_type}/{task_name}...")
    df = pd.read_pickle(f"{data_dir}/{task_type}/{task_name}.pkl")

    if only_correct:
        df = df[df["is_correct_baseline"]].reset_index(drop=True)
    # Dropping empty prompt sources. This is an artifact of saving and reloading inputs
    df = df[~df["prompt_source"].apply(lambda x: isinstance(x, float))].reset_index(drop=True)
    # Dropping prompt sources with \n. pandas read_pickle is not able to handle them properly and drops the rest of the input.
    df = df[~df["prompt_source"].str.contains('\n')].reset_index(drop=True)
    # After manual inspection, this example seems to have tokenization issues. 0Dropping.
    if task_name == "star_constellation":
        df = df[~df["prompt_source"].str.contains("service")].reset_index(drop=True)
    elif task_name == "object_superclass":
        df = df[~df["prompt_source"].str.contains("Swainson ’ s hawk and the prairie")].reset_index(drop=True)
    
    def tokenize_and_count(text):
        encoding = mt.tokenizer.tokenize(text)
        return len(encoding)
    df['token_count'] = df['prompt_source'].apply(tokenize_and_count)
    df = df[df['token_count'] > df['position_source']].reset_index(drop=True)
    print(f"\tNumber of samples: {len(df)}")

    # BATCHED
    batch = []
    for _, row in tqdm(df.iterrows()):
        for layer_source in range(mt.num_layers-1):
            for layer_target in range(mt.num_layers-1):
                item = dict(row)
                item.update({
                    "layer_source": layer_source,
                    "layer_target": layer_target,
                })
                batch.append(item)
    experiment_df = pd.DataFrame.from_records(batch)# 将列表转换为DataFrame,即将列表中的字典转换为DataFrame的行
    
    if n_samples > 0 and n_samples<len(experiment_df):
        experiment_df = experiment_df.sample(n=n_samples, replace=False, random_state=42).reset_index(drop=True)# 抽样

    print(f"\tNumber of datapoints for patching experiment: {len(experiment_df)}")
    

    # eval_results = evaluate_attriburte_exraction_batch(mt, experiment_df, batch_size=batch_size, is_icl=is_icl)
    
    eval_results = evaluate_attriburte_exraction_batch_multi_patch(mt, experiment_df, batch_size=batch_size, is_icl=is_icl)
    
    # eval_results = evaluate_attriburte_exraction_batch_llama3_multi_patch (mt, experiment_df, batch_size=batch_size, is_icl=is_icl)

    results_df = experiment_df.head(len(eval_results["is_correct_patched"]))
    for key, value in eval_results.items():
        results_df[key] = list(value)

    if save_output:
        fdir_out = f"{output_dir}/{task_type}"
        if not os.path.exists(fdir_out):
            os.makedirs(fdir_out)
        results_df.to_csv(f"{fdir_out}/{task_name}_only_correct_{only_correct}.tsv", sep="\t")
        results_df.to_pickle(f"{fdir_out}/{task_name}_only_correct_{only_correct}.pkl")
        

    return results_df

# for task_type in ["commonsense", "factual"]:
# for task_type in ["commonsense"]:
for task_type in ["factual"]:
    for fname in tqdm(os.listdir(f"./preprocessed_data/pythia/{task_type}")):
        if fname.endswith('.pkl'):
            task_name = fname[:-4]
        else:
            continue
        print(f"Processing {fname}...")
        run_experiment(task_type, task_name,
                        data_dir="./preprocessed_data/pythia",
                        output_dir=f"./multi_patch_output/pythia/3_patch",
                        batch_size=512,
                        is_icl=False,
                        only_correct=False,
                        replace=False,
                        )