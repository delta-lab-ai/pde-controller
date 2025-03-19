#!/bin/bash
python --version

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")


### --------- Set Parameters --------- ###


##### model parameters #####

tokenizer_path=models/base_model/MathCoder2-DeepSeekMath-7B
context_length=4096

##### Data parameters #####
input_file_paths=(
    datasets/unprocessed/sft/heat_nc1_train
    datasets/unprocessed/sft/heat_nc2_train
    datasets/unprocessed/sft/heat_nc3_train
    datasets/unprocessed/sft/wave_nc1_train
    datasets/unprocessed/sft/wave_nc2_train
    datasets/unprocessed/sft/wave_nc3_train
)

# ["to_python_no_STL", "to_STL", "to_python_GT_STL", "to_python_given_STL"]
prompt_format=to_STL

out_file_path=datasets/sft/${prompt_format}

### To set for "to_python_given_STL"  ####
max_samples=-1 # -1 to test all datapoints.
TRANSLATOR_DIR=models/translator 


######################################
### --------- Data processing --------- ###
######################################


if [ "$prompt_format" = "to_python_given_STL" ]; then
    ### --------- prompt the model for its predictions and add this to the training data. --------- ###


    ### --------- First step: train the model on the original data. --------- ###
    prompt_format=to_STL

    few_shot_number=2  

    # join the input file paths into a single string
    input_file_paths_str=$(IFS=" "; echo "${input_file_paths[*]}")

    predictions_output_dir=$out_file_path/train_eval
    # obtain the model's predictions for the second step
    "$SCRIPT_DIR/run_train_testing.sh" "$input_file_paths_str" "$prompt_format" "$few_shot_number" $TRANSLATOR_DIR $predictions_output_dir $max_samples
    # the second step continues below
    prompt_format=to_python_given_STL

    input_file_paths=(
        $predictions_output_dir/to_STL/
    )
    ### --------- End of first step --------- ###
fi


python train/scripts/tokenize_data.py --paths "${input_file_paths[@]}" --out_file_path ${out_file_path} --model_path $tokenizer_path --sft --prompt_format $prompt_format
python train/scripts/group_text.py --max_len $context_length --out_file_path ${out_file_path} --model_path $tokenizer_path --sft --no_grouping --no_padding --balance 0.05 0.22 0.23 0.05 0.22 0.23 --total 128000


######################################
### --------- Training --------- ###
######################################

echo Starting training...

bash train/scripts/train.sh
