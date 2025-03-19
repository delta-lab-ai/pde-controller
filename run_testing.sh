#!/bin/bash

eval "$(conda shell.bash hook)"

PROJ_DIR=/localhome/mms43/scratch/mathcoder2/MathCoder2
MODELS="/localhome/mms43/scratch/mathcoder2/outputs"
BASE_MODEL_PATH=/localhome/mms43/scratch/mathcoder2/model_ckpts/MathCoder2-DeepSeekMath-7B
OUTPUT_DIR=/localhome/mms43/scratch/mathcoder2/outputs_new

########
DIR_TRANSLATOR=$MODELS/30233_ds_to_STL_s16800/checkpoint-step-3000/merged_model    
# DIR_CODER=$MODELS/30236_ds_to_python_given_STL_s16800/checkpoint-step-3000/merged_model          ### coder (model to evaluate)
# DIR_CODER=$BASE_MODEL_PATH                                                                           ### MathCoder2-DeepSeekMath-7B                             
# DIR_CODER=$MODELS/30234_ds_to_python_no_STL_s16800/checkpoint-step-6000/merged_model             ### trained baseline
# DIR_CODER==$MODELS/30235_ds_to_python_GT_STL_s16800/checkpoint-step-3000/merged_model            ### oracle
DIR_CODER=$MODELS/10136_ds_to_python_misaligned_s16800/checkpoint-step-1500/merged_model         ### finetuned coder (model to evaluate)
DIR_CONTROLLER=$MODELS/10138_ds_DPO__s16800/checkpoint-step-16000/policy/merged_model            ### controller





gpus="1"


use_openai=
# use_openai='gpt-4o'
# use_openai='o1-mini'


cd $PROJ_DIR


if [ -n "$use_openai" ]; then
    EVAL_DIR=$OUTPUT_DIR/$use_openai/eval
else
    EVAL_DIR=${OUTPUT_DIR}/eval 
fi


bash test/scripts/test_pdecontrol.sh $EVAL_DIR $gpus $DIR_TRANSLATOR $DIR_CODER $DIR_CONTROLLER $use_openai
