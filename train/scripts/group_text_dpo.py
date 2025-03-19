#!/usr/bin/env python3

import os
import json
import random
import numpy as np
from utils.loader import Processor
from datasets import load_dataset, concatenate_datasets
import argparse
from transformers import (
    AutoTokenizer, 
)

os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))

def find_common_keywords_update_outfile(file_paths, outfile, ignore_keywords=None):
    if ignore_keywords is None:
        ignore_keywords = []
    # extract the basenames from the filenames
    basenames = [os.path.basename(file_name) for file_name in file_paths]
    # split the basenames into words by underscores, ignoring the file extension
    words_list = [set(os.path.splitext(basename)[0].split('_')) for basename in basenames]
    # filter the keywords if any
    words_list = [words - set(ignore_keywords) for words in words_list]
    # common words in all basenames
    common_words = set.intersection(*words_list)
    # modify the out_file basename to include these common words
    out_file_basename = os.path.splitext(outfile)[0]  # remove the extension from outfile
    if common_words:
        out_file_basename += "_" + "_".join(sorted(common_words))
    out_file_basename += os.path.splitext(outfile)[1]  # add the original extension back
    return out_file_basename

def balance_datasets(datasets, balance, total=None, dataset_names=None):
    """Balance the datasets based on the balance values. Assumes each dataset was pre-shuffled.
    # Args:
        datasets: list of datasets to balance.
        balance: list(int). The percentage of data to keep from each dataset. The sum of the list must be 1.
        total: int. The total number of data to keep.
        dataset_names: list(str). The names of the datasets.
     ---
    All of these (except point 4) assume `total < sum([len(dataset) for dataset in datasets])`.

    1. `Total = None, Balance = [1]` (**default**): Keep everything.
    2. `Total = int, Balance = [1]`: Keep total number of data. Will first sample uniformly from all datasets, then sample from the remaining datasets based on the total.
    3. `Total = None, len(Balance) > 1`: Keep all data for the smallest dataset and the rest will determined based of balance.
    4. `Total = int, len(Balance) > 1`: Keep total number of data, and balance based on the balance values. This will may double-sample random datapoints from datasets that are too small.
    """
    assert sum(balance) == 1, "The balance values must sum to 1."
    if len(balance) > 1:
        assert len(datasets) == len(balance), "The number of datasets and `balance` values must be the same."
        assert dataset_names is None or len(datasets) == len(dataset_names), "The number of datasets and `dataset_names` must be the same."
    
    balanced_datasets = []
    if total is None:
        if balance == [1]:
            # 1. Keep everything
            return datasets
        else:
            # 3. Keep all data for the smallest dataset and balance the rest
            min_i, min_dataset = min(enumerate(datasets), key=lambda x: len(x[1]))
            total_size = len(min_dataset) / balance[min_i]

            for i, (dataset, proportion) in enumerate(zip(datasets, balance)):
                num_to_keep = int(total_size * proportion)
                dataset_name = dataset_names[i] if dataset_names else f"Dataset {i+1}"
                print(f"Sampling {num_to_keep} datapoints from {dataset_name} of size {len(dataset)}")
                balanced_datasets.append(dataset.select(range(num_to_keep)))
    else:
        if balance == [1]:
            # 2. Keep total number of data. Since assumes data was pre-shuffled, we can just take the first `total` number of data.
            balanced_datasets = [dataset.select(range(total)) for dataset in datasets]
        else:
            # 4. Keep total number of data, balance based on balance values
            for i, (dataset, proportion) in enumerate(zip(datasets, balance)):
                num_to_keep = int(total * proportion)
                dataset_name = dataset_names[i] if dataset_names else f"Dataset {i+1}"
                temp_dataset = None
                if len(dataset) < num_to_keep:
                    full_repeats = num_to_keep // len(dataset)
                    remainder = num_to_keep % len(dataset)
                    for _ in range(full_repeats):
                        if temp_dataset is None:
                            temp_dataset = dataset
                        else:
                            temp_dataset = concatenate_datasets([temp_dataset, dataset])
                    indices_to_keep = random.sample(range(len(dataset)), remainder)
                    print(f"Sampling {num_to_keep} datapoints from {dataset_name} of size {len(dataset)} (with {full_repeats} replications for all points and random sampling for the remainder)")
                    temp_dataset = concatenate_datasets([temp_dataset, dataset.select(indices_to_keep)])
                    balanced_datasets.append(temp_dataset)
                else:
                    print(f"Sampling {num_to_keep} datapoints from {dataset_name} of size {len(dataset)}")
                    indices = random.sample(range(len(dataset)), num_to_keep)
                    balanced_datasets.append(dataset.select(indices))
    return balanced_datasets



def group_text(tokenizer, out_file, max_len, sft=False, no_grouping=False, no_padding=False, args=None):
    seed = 3407
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    train_file_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "file_names_to-be-trained.json")
    with open(train_file_config, "r") as f:
        train_files = json.load(f)
        
    out_file = find_common_keywords_update_outfile(train_files, out_file)
    train_sets = []
    for file in train_files:
        print("loading file:", file)
        _dataset = load_dataset(file.split(".")[-1] if file.split(".")[-1] != "jsonl" else "json", data_files=file, split='train')
        
        # shuffle the dataset. this is important for the balance_datasets function
        _dataset = _dataset.shuffle(seed=seed)
        train_sets.append(_dataset)

    train_sets = balance_datasets(train_sets, args.balance, args.total, train_files)
    
    lengths = np.array([_set.shape[0] for _set in train_sets])
    print(f"Data Lengths: {lengths}")
    train_sets = concatenate_datasets(train_sets)
    print(f"Total Length: {train_sets.shape[0]}")
    process_batch_size = min(1000, len(train_sets))
    
    processor = Processor()
    train_sets = train_sets.shuffle(seed=seed)
    column_names = list(train_sets.features)

    if no_grouping:
        if no_padding:
            # only truncate
            train_sets = train_sets.map(
                processor.truncate,
                fn_kwargs={
                    "max_len": max_len
                },
                batched=True,
                load_from_cache_file=False,
                batch_size=process_batch_size,
                num_proc=96,
                desc=f"Checking texts for chunks > {max_len}. Will truncate. No padding. No grouping.",
            )
        else:
            # truncate and add padding
            train_sets = train_sets.map(
                processor.truncate_and_add_padding,
                fn_kwargs={
                    "tokenizer": tokenizer,
                    "max_len": max_len
                },
                batched=True,
                load_from_cache_file=False,
                batch_size=process_batch_size,
                num_proc=96,
                desc=f"Checking texts for chunks > {max_len}. Will pad and truncate. No grouping.",
            )
    else:
        if sft:
            # group texts for seq-to-seq
            raise NotImplementedError("Seq-to-seq grouping used to be implemented but this code needs to be checked before using again.")
        else:
            # pretraining: the original grouping function
            train_sets = train_sets.map(
                processor.group_texts,
                fn_kwargs={
                    "tokenizer": tokenizer, 
                    "max_len": max_len
                },
                batched=True,
                load_from_cache_file=False,
                remove_columns=column_names,
                batch_size=process_batch_size,
                num_proc=96,
                desc=f"Grouping texts in chunks of {max_len}",
            )
    train_sets.to_parquet(out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", action="store_true", help="Run for supervised fine-tuning data")
    parser.add_argument("--max_len", type=int, default=4096, help="Max context length (or the length to use for truncation and padding if applicable)")
    parser.add_argument("--out_file_path", type=str, default=None, help="Path to output grouped parquet file") 
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B", help="Model path")
    parser.add_argument("--no_grouping", action="store_true", help="Whether to not group smaller sequences and truncate longer ones when grouping.")
    parser.add_argument("--no_padding", action="store_true", help="Whether to not pad shorter sequences.")
    parser.add_argument("--balance", type=float, nargs="+", default=[1], help="The percentage of data to keep from each dataset. The sum of the list must be 1.")
    parser.add_argument("--total", type=int, default=None, help="The total number of data to keep.")

    args = parser.parse_args()

    sft = args.sft
    model_path = args.model_path
    max_len = args.max_len # context length
    out_file = f"{args.out_file_path}/{'not-' if args.no_grouping else ''}grouped_MaxContext{max_len}.parquet"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    group_text(tokenizer, out_file, max_len, sft, no_grouping=args.no_grouping, no_padding=args.no_padding, args=args)