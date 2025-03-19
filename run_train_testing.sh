#!/bin/bash

dataset_list=$1
prompt_format=$2
few_shot_number=$3
checkpoint_dir=$4
out_file_path=$5
max_samples=$6

# Split the dataset_list into an array
IFS=' ' read -r -a datasets <<< "$dataset_list"


for dataset in "${datasets[@]}"
do
    CMD="CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false python test/PDEcontrol/evaluation/infer/run_1d_pdecontrol_eval_train.py \
    --data_dir $dataset \
    --save_dir $out_file_path \
    --use_vllm \
    --model_name_or_path $checkpoint_dir \
    --tokenizer_name_or_path $checkpoint_dir \
    --eval_batch_size 1 \
    --temperature 0.0 \
    --prompt_format few_shot \
    --few_shot_number $few_shot_number \
    --few_shot_prompt $prompt_format"

    if [[ $dataset == *"heat"* ]]; then
        CMD="$CMD --prompt_dataset heat"
    elif [[ $dataset == *"wave"* ]]; then    
        CMD="$CMD --prompt_dataset wave"
    fi

    if [ $max_samples -gt 0 ]; then
        CMD="$CMD --max_num_examples $max_samples"
    fi

    echo $CMD
    eval $CMD
done


