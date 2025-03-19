# Supervised Finetuning and Direct Preference Optimization

This directory contains the code for SFT and DPO.

## Installation

Follow the instructions on [the main readme](../README.md)

## Training

### Step 1: save the paths of the jsonl files to be tokenized

Modify `input_file_paths` in `run_training.sh`. Exch element of `input_file_paths` should be the path to a directory that contains `.jsonl` files with the texts to be trainined under the key `"text"`. For example:
```
    datasets/unprocessed/sft/one_d_heat_train
    datasets/unprocessed/sft/one_d_wave_train
```

You can modify `out_file_path` in `run_training.sh` to change the output directory for the parquet files. Then run:

```shell
bash run_training.sh
```
#### Tokenization
 This script will automatically tokenize the file when calling `train/scripts/tokenize_data.py`.  The paths for the output tokenized parquet files in automatically be written to the file `file_names_to-be-trained.json`. Each element should be a path to a directory that contains parquet files (created from tokenization). 

#### Grouping the texts into given context length

You can modify `max_len` in `run_training.sh` to change the context length. This context length should equal the context length you wish to use in training. `train/scripts/group_text.py` will get called. This outputs a single parquet file containing token indexes grouped into the given context length.

### Step 2: train the model

For example:

You can modify the training configs in `scripts/train.py`. The example script runs on a single node with 4 A100 GPUs. You may need to modify the script so that it runs on your cluster. Run the following command (it is also included in the `run_training.sh` script)

```shell
bash train/scripts/train.sh
```

This script can train the model with validation, or validation can be turned off by simply running:
```sh
train
exit_status=$?
```