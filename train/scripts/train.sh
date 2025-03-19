python --version

export NCCL_DEBUG=WARN

export NCCL_IB_TIMEOUT=22   
export NCCL_IB_RETRY_CNT=13 
export NCCL_IB_AR_THRESHOLD=0

wandb login $WANDB_TOKEN

# Constants for paths
OUTS="outputs"
MODELS="models"
DATA="datasets/sft"

# variables
base_run_name=test
max_steps=4 
per_gpu_batch_size=8 
accum_grad=8
epoch_size=2

save_meged_model=True

# ["to_python_no_STL", "to_STL", "to_python_GT_STL", "to_python_given_STL", "to_python_misaligned""]
prompt_format=to_STL


### specify path to base models for which an adapter will be attached.
base_model="$MODELS/MathCoder2-DeepSeekMath-7B"


### Distributed settings:
# export MASTER_ADDR=$(hostname -i)
# export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export GPUPerNode=2
export CUDA_VISIBLE_DEVICES=2,3



run_name=${base_run_name}_${prompt_format}_s${max_steps}

datafile_name=${prompt_format}/not-grouped_MaxContext4096_tokenized_train.parquet

if [ "$prompt_format" = "to_python_given_STL" ]; then
  datafile_name=${prompt_format}/not-grouped_MaxContext4096_STL_predictions_to_tokenized_train.parquet
elif [ "$prompt_format" = "to_python_misaligned" ]; then
  DATA="/localhome/mms43/scratch/mathcoder2/datasets/pdecontrol/dpo"
  datafile_name=${prompt_format}/not-grouped_MaxContext4096_misaligned_aligned_tokenized_train.parquet
fi



find_latest_checkpoint() {
  local checkpoint_dir=$1
  latest_checkpoint=$(ls -d ${checkpoint_dir}/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
  echo $latest_checkpoint
}


train() {
  # directory where checkpoints are stored
  CHECKPOINT_DIR=$OUTS/$run_name

  export WANDB_DIR=$OUTS/$run_name
  export WANDB_RUN_ID=$run_name

  cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES OMP_NUM_THREADS=1 torchrun --nnodes $WORLD_SIZE --node_rank $RANK --nproc_per_node $GPUPerNode --rdzv-backend=c10d --rdzv-endpoint=localhost:0 train/train_finetune.py \
  --ddp_timeout 360000 \
  --train_parquet_file ${DATA}/${datafile_name} \
  --run_name $run_name \
  --output_dir $OUTS/${run_name} \
  --no_timestamps \
  --dataloader_num_workers 2 \
  --max_len 4096 \
  --max_steps $max_steps \
  --num_train_epochs -1 \
  --save_steps 5 \
  --save_total_limit 2 \
  --step_save_interval $epoch_size \
  --warmup_steps 50 \
  --logging_steps 10 \
  --learning_rate 4e-5 \
  --weight_decay 0.1 \
  --lr_scheduler_type cosine \
  --per_device_train_batch_size $per_gpu_batch_size \
  --gradient_accumulation_steps $accum_grad \
  --seed 3407 \
  --deepspeed train/config/deepspeed.json \
  --bf16 \
  --stream \
  --do_train \
  --gradient_checkpointing \
  --report_to wandb \
  --lora_r 64 \
  --lora_alpha 256 \
  --lora_dropout 0.1 \
  --model_cfg $base_model"

  if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "Will try to load checkpoint from $LATEST_CHECKPOINT"
    export WANDB_RESUME="auto"
    cmd="$cmd --resume_from $LATEST_CHECKPOINT"
  fi

  if [ "$external_validation" == "True" ]; then
    cmd="$cmd --external_validation"
  fi

  eval $cmd
}



############## Training loop with validation ###################
################################################################

validate() {
  local checkpoint_dir=$1
  python train/scripts/merge_model.py --adapter_path $checkpoint_dir --output_dir $checkpoint_dir --model_path $base_model --gpus $CUDA_VISIBLE_DEVICES
  echo "validating"
  python train/validate.py --checkpoint_dir $checkpoint_dir --base_model $base_model --cuda_visible_devices $CUDA_VISIBLE_DEVICES --validation_data_dir /localhome/mms43/scratch/mathcoder2/MathCoder2/test/PDEcontrol/validation_data --wandb_dir $OUTS/${run_name}
}



validate_interval=$epoch_size
iterations=$((max_steps / validate_interval))
epoch_size=0
external_validation=True
for ((r=0; r<iterations; r++)); do
  epoch_size=$((epoch_size + validate_interval))
  if [ $r -eq 0 ]; then
    train
  else
    last_epoch=$((epoch_size - validate_interval))
    LATEST_CHECKPOINT=$CHECKPOINT_DIR/backups/checkpoint-$last_epoch
    echo "LATEST_CHECKPOINT: $LATEST_CHECKPOINT"
    train 
  fi
  exit_status=$?
  validate $CHECKPOINT_DIR/checkpoint-step-$epoch_size
done



##### just train.
# train
# exit_status=$?


if [ $exit_status -eq 0 ]; then
  echo "Training completed successfully."
  if [ "$save_meged_model" = "True" ]; then
    python train/scripts/merge_model.py --adapter_path $CHECKPOINT_DIR/final --output_dir $OUTS/${run_name} --model_path $base_model
  fi
fi