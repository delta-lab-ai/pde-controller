import argparse
import json
import os
import sys
import wandb
import concurrent.futures

current_dir = os.path.dirname(os.path.realpath(__file__))
merge_path = os.path.join(current_dir, "scripts")
sys.path.append(merge_path)
import merge_model

eval_path = os.path.join(current_dir, "../test/PDEcontrol/evaluation/infer")
sys.path.append(eval_path)
import run_1d_pdecontrol_eval_full as run_1d_pdecontrol_eval



def create_merge_args(args):
    class merge_Args:
        model_path = args.base_model
        output_dir = f"{args.checkpoint_dir}"
        adapter_path = args.checkpoint_dir
        gpus = args.cuda_visible_devices
    return merge_Args()


def create_eval_args(args, data_type, p_format, validation_dir, save_directory, shots=2):
    class eval_Args:
        data_dir = validation_dir
        python_key = "python"
        stl_key = "sstl"
        nl_key = "nl"
        robustness_key = "robustness"
        max_num_examples = args.valid_num_examples
        save_dir = f"{save_directory}/{shots}_shots"
        model_name_or_path_coder = f"{args.checkpoint_dir}/merged_model"
        tokenizer_name_or_path_coder = args.base_model
        eval_batch_size = len(args.cuda_visible_devices) 
        load_in_8bits = False
        gptq= False
        use_vllm = True
        load_in_half = False
        infer_on_train_set=True
        n_subsets=1
        subset_id=0
        temperature = 0
        repeat_id_start = 0
        n_repeat_sampling = 1
        prompt_format = "few_shot"
        few_shot_prompt = p_format
        prompt_dataset = data_type
        few_shot_number = shots
        answer_extraction_fn=None
        eval_perplexity=True
        eval_robustness = True
        eval_edit_distance = True
        eval_iou = True
        load_from_file=False
        skip_existing_scores=False
        gpus = args.cuda_visible_devices      
        seed = None
        use_openai = False
    return eval_Args()


def delete_merged_model(checkpoint_dir):
    os.system(f"rm -rf {checkpoint_dir}/merged_model")

def log_to_wandb(results_dir, args):
    for directory in os.listdir(results_dir):
        eval_shots_dir = os.path.join(results_dir, directory)
        if os.path.isdir(eval_shots_dir):
            for eval_method_subdir in os.listdir(eval_shots_dir):
                eval_method_subdir_path = os.path.join(eval_shots_dir, eval_method_subdir)
                if os.path.isdir(eval_method_subdir_path):
                    metrics_file = os.path.join(eval_method_subdir_path, f"metrics.{args.valid_num_examples}.json")
                    if os.path.isfile(metrics_file):
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                        
                        few_shot_prompt = metrics.get("few_shot_prompt", "")
                        few_shot_number = metrics.get("few_shot_number", None)
                        
                        for metric, value in metrics.items():
                            if metric not in ["few_shot_prompt", "few_shot_number"]:
                                wandb.log({
                                    f"validation_{few_shot_prompt}_{few_shot_number}shots/{metric}": value
                                })


def validate_model(args):
    wandb.init(project=args.project_name, resume=True, dir=args.wandb_dir)

    merge_args = create_merge_args(args)
    merge_model.main(merge_args)

    for dataset in os.listdir(args.validation_data_dir):
        dataset_path = os.path.join(args.validation_data_dir, dataset)
        if os.path.isdir(dataset_path):
            for shots in [0, 2]:
                for p_format in ["to_python_no_STL"]:
                    if 'heat' in dataset:
                        data_type = "CoTOneDHeat"
                    elif 'wave' in dataset:
                        data_type = "CoTOneDWave"

                    save_dir = f"{args.checkpoint_dir}/validation"
                    eval_args = create_eval_args(args, data_type, p_format, dataset_path, save_dir, shots)
                    run_1d_pdecontrol_eval.main(eval_args)

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_1d_pdecontrol_eval.run_main, eval_args)
                        try:
                            future.result(timeout=900)  # Set timeout in seconds
                        except concurrent.futures.TimeoutError:
                            print(f"Timeout occurred for dataset: {dataset}, shots: {shots}, p_format: {p_format}")



    delete_merged_model(args.checkpoint_dir)
    log_to_wandb(save_dir, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--cuda_visible_devices', type=str, required=True)
    parser.add_argument('--valid_num_examples', type=int, default=8)
    parser.add_argument('--validation_data_dir', type=str, required=True)
    parser.add_argument('--project_name', type=str, default="huggingface")
    parser.add_argument('--wandb_dir', type=str, required=True)
    

    args = parser.parse_args()

    # vllm may freeze on multi-gpu
    args.cuda_visible_devices = args.cuda_visible_devices.split(',')[0]

    validate_model(args)