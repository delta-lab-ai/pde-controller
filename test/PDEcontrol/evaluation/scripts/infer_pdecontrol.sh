eval "$(conda shell.bash hook)"

set -e

conda activate trainenv
echo "activated conda env:" $CONDA_DEFAULT_ENV
python --version



dataset=$1
out_dir=$2
few_shot_number=$3
prompt_format=$4
max_samples=$5
gpus=$6
translator_path=${7}
coder_path=${8}
controller_path=${9}
use_openai=${10}

save_dir=$out_dir/${dataset}_shots=${few_shot_number}


CMD="CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false python test/PDEcontrol/evaluation/infer/run_1d_pdecontrol_eval_full.py \
--data_dir \"$(dirname $0)/../../test_data/${dataset}\" \
--save_dir $save_dir \
--use_vllm \
--model_name_or_path_translator $translator_path \
--tokenizer_name_or_path_translator $translator_path \
--model_name_or_path_coder $coder_path \
--tokenizer_name_or_path_coder $coder_path \
--model_name_or_path_controller $controller_path \
--tokenizer_name_or_path_controller $controller_path \
--eval_batch_size 1 \
--temperature 0.2 \
--seed 0
--n_repeat_sampling 3
--prompt_format few_shot \
--prompt_dataset CoTOneDCombined \
--few_shot_number $few_shot_number \
--few_shot_prompt $prompt_format \
--eval_robustness
--eval_iou
--eval_edit_distance"

if [ $max_samples -gt 0 ]; then
    CMD="$CMD --max_num_examples $max_samples"
fi

if [ "$gpus" != -1 ]; then
    CMD="$CMD --gpus $gpus"
fi

if [ -n "$use_openai" ]; then
    CMD="$CMD --use_openai $use_openai"
else
    CMD="$CMD --eval_perplexity"
fi

echo $CMD
eval $CMD

exit 1