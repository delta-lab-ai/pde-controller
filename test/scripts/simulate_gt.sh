DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd $DIR/../..



declare -a datasets=(
   "heat_nc1"
   "heat_nc2"
   "heat_nc3"
   "wave_nc1"
   "wave_nc2"
   "wave_nc3"
)


## set the max samples to be -1 to use all samples.
max_samples=512
skip_existing=False       # True or False
load_from_file=False      # True or False
gpus=1


# Calculate the total number of iterations
total_iterations=$(( ${#datasets[@]} ))
current_iteration=0
# Record the start time
start_time=$(date +%s)

for dataset in "${datasets[@]}"
do
   echo "    PROGRESS: $current_iteration/$total_iterations"
   echo "    Evaluating $dataset" | tee -a $log_file

   if [[ $prompt_format == "to_STL" ]]; then
      eval_robustness=False
   fi

   CMD="CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false python test/PDEcontrol/evaluation/infer/simulate_gt.py \
   --data_dir \"$DIR/../../test/PDEcontrol/test_data/${dataset}\" \
   --eval_batch_size 1 \
   --eval_robustness"

   if [ $max_samples -gt 0 ]; then
      CMD="$CMD --max_num_examples $max_samples"
   fi

   if [ "$load_from_file" = "True" ]; then
      CMD="$CMD --load_from_file"
   fi

   if [ -n "$gpus" ]; then
      CMD="$CMD --gpus $gpus"
   fi


   echo $CMD
   eval $CMD



   echo "    Done evaluating $dataset with $few_shot few shot examples and format=$prompt_format"
   current_iteration=$((current_iteration + 1))
   echo "    PROGRESS: $current_iteration/$total_iterations"
   # Calculate elapsed time
   current_time=$(date +%s)
   elapsed_time=$((current_time - start_time))
   # Estimate remaining time
   average_time_per_iteration=$((elapsed_time / current_iteration))
   remaining_iterations=$((total_iterations - current_iteration))
   estimated_remaining_time=$((remaining_iterations * average_time_per_iteration))
   hours=$((estimated_remaining_time / 3600))
   minutes=$(( (estimated_remaining_time % 3600) / 60 ))
   seconds=$((estimated_remaining_time % 60))
   echo "    Estimated remaining time: $hours hours, $minutes minutes, $seconds seconds"
done
