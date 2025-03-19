#!/bin/bash


# DeepSeekMath-7B mothcoder2 version
MODEL_PATH=models/base_model/MathCoder2-DeepSeekMath-7B
OUTPUT_DIR=outputs/dpo

##### model parameters #####
context_length=4096

##### Data parameters #####
input_file_paths=(
    datasets/unprocessed/dpo/dpo_d0_train
    datasets/unprocessed/dpo/dpo_d1_train
    datasets/unprocessed/dpo/dpo_d2_train
)
out_file_path=datasets/dpo



python train/scripts/tokenize_data_dpo.py --paths "${input_file_paths[@]}" --out_file_path ${out_file_path}/train --model_path $MODEL_PATH
python train/scripts/group_text.py --max_len $context_length --out_file_path ${out_file_path}/train --model_path $MODEL_PATH --no_grouping --no_padding --balance 1 --dpo


echo Starting training...

bash train/scripts/train_dpo.sh