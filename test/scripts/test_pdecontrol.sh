out_dir=${1}
gpus=${2}
translator_path=${3}
coder_path=${4}
controller_path=${5}
use_openai=${6}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

declare -a datasets=(
   "heat_nc1_512"
   "heat_nc2_512"
   "heat_nc3_512"
   "wave_nc1_512"
   "wave_nc2_512"
   "wave_nc3_512"
   # "dpo_manual_test"
)

declare -a prompt_formats=(
   # to_python_no_STL
   # to_python_direct_with_sstl_cot
   to_python_two_step
   # to_STL
   # full_pipeline
   # reasoning_only
)

declare -a few_shot_numbers=(
   0
   2
)


## set the max samples to be -1 to use all samples.
max_samples=4
gpus=$gpus
# gpus=0

if [[ -n $use_openai ]]; then
   declare -a few_shot_numbers=(
      2
   )
fi

# appends to the log file
log_file=${out_dir}/eval_output.log

# Calculate the total number of iterations
total_iterations=$(( ${#datasets[@]} * ${#prompt_formats[@]} * ${#few_shot_numbers[@]}))
current_iteration=0
# Record the start time
start_time=$(date +%s)

for dataset in "${datasets[@]}"
do
   for prompt_format in "${prompt_formats[@]}"
   do
      for few_shot in "${few_shot_numbers[@]}"
      do
         echo "    PROGRESS: $current_iteration/$total_iterations" | tee -a $log_file
         echo "    Evaluating $dataset with $few_shot few shot examples and format=$prompt_format" | tee -a $log_file

         echo dataset: $dataset
         echo out_dir: $out_dir
         echo shots: $few_shot
         echo translator_path: $translator_path
         echo coder_path: $coder_path
         echo controller_path: $controller_path
         echo prompt_format: $prompt_format
         echo max_samples: $max_samples
         echo gpus: $gpus
         echo use_openai: $use_openai


         bash test/PDEcontrol/evaluation/scripts/infer_pdecontrol.sh $dataset $out_dir $few_shot $prompt_format $max_samples $gpus $translator_path $coder_path $controller_path $use_openai | tee -a $log_file

         echo "    Done evaluating $dataset with $few_shot few shot examples and format=$prompt_format" | tee -a $log_file
         current_iteration=$((current_iteration + 1))
         echo "    PROGRESS: $current_iteration/$total_iterations" | tee -a $log_file
         # Calculate elapsed time
         current_time=$(date +%s)
         elapsed_time=$((current_time - start_time))
         # Estimate remaining time
         average_time_per_iteration=$((elapsed_time / current_iteration))
         remaining_iterations=$((total_iterations - current_iteration))
         estimated_remaining_time=$((remaining_iterations * average_time_per_iteration))
         # Convert estimated remaining time to human-readable format
         hours=$((estimated_remaining_time / 3600))
         minutes=$(( (estimated_remaining_time % 3600) / 60 ))
         seconds=$((estimated_remaining_time % 60))
         elapsed_hours=$((elapsed_time / 3600))
         elapsed_minutes=$(( (elapsed_time % 3600) / 60 ))
         elapsed_seconds=$((elapsed_time % 60))
         echo "    Elapsed time: $elapsed_hours hours, $elapsed_minutes minutes, $elapsed_seconds seconds" | tee -a $log_file
         echo "    Estimated remaining time: $hours hours, $minutes minutes, $seconds seconds" | tee -a $log_file
      done
   done
done

# # Initialize max_shots with the first element of the array
# max_shots=${few_shot_numbers[0]}

# # Iterate through the array to find the maximum value
# for num in "${few_shot_numbers[@]}"; do
#     # Skip commented out values
#     if [[ $num =~ ^[0-9]+$ ]]; then
#         if (( num > max_shots )); then
#             max_shots=$num
#         fi
#     fi
# done

# echo "Maximum shots: $max_shots"

# CMD="python /localhome/mms43/scratch/mathcoder2/MathCoder2/test/scripts/read_result.py --in_dir $out_dir --subset_id $max_samples --shots $max_shots --seeds ${seeds[@]}"

# CMD="$CMD --eval_methods ${prompt_formats[@]}"
# # if [[ $prompt_format == "to_STL" ]]; then
# #     CMD="$CMD --eval_methods to_STL"
# # fi

# eval $CMD