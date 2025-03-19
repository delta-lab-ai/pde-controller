import json
from argparse import ArgumentParser
import os
from glob import glob

import numpy as np

def read_json(in_file):
    with open(in_file, "r", encoding="utf-8") as f:
        return json.load(f)


def per_eval_method_read_result(metrics, datasets, in_dir, eval_method, subset_id=None, shots=3, seeds=[-1]):
    # Initialize the output text
    text = ""
    for metric in metrics:
        # Initialize a dictionary to store results
        max_shots = shots
        # results = {i: ["n/a"] * len(datasets) for i in range(max_shots + 1)}
        ## for each shot, there is a dataset. Each dataset contains a list to track the scores for each seed.
        results = {i: {dataset: [] for dataset in datasets} for i in range(max_shots + 1)}
        
        # Track dataset columns with valid entries
        valid_columns = {dataset: False for dataset in datasets}
        
        # Iterate over the directories again to populate results
        for dirname in os.listdir(in_dir):
            if "shots=" in dirname and "seed=" in dirname:
                # Extract shot number and seed number
                shot_num = int(dirname.split("shots=")[1].split("_")[0])
                seed_num = int(dirname.split("seed=")[1].split("_")[0])

                if seed_num not in seeds:
                    continue
                
                # Determine the dataset and column based on the directory name
                dataset = dirname.split("_shots=")[0]
                
                # Read the metrics.json file
                in_file = os.path.join(in_dir, dirname, eval_method, "metrics.json")
                if subset_id is not None:
                    in_file = os.path.join(in_dir, dirname, eval_method, f"metrics.{subset_id}.json")
                if os.path.exists(in_file):
                    data = read_json(in_file)
                    try:
                        results[shot_num][dataset].append(data[metric])
                        valid_columns[dataset] = True
                    except KeyError:
                        pass
        
        # Ensure all datasets have the same number of scores for each shot number (excluding the empty)
        for shot_num in range(max_shots + 1):
            lengths = [len(results[shot_num][dataset]) for dataset in datasets if valid_columns[dataset]]
            if lengths and not all(length == lengths[0] for length in lengths):
                raise ValueError(f"Inconsistent number of scores for shot number {shot_num}: {lengths}")
        

        # Filter out columns that are entirely empty
        filtered_datasets = []
        filtered_columns = []
        for dataset in datasets:
            if valid_columns[dataset]:
                filtered_datasets.append(dataset)
                filtered_columns.append(dataset)
        
        # Initialize the table header
        header = f"## Metric: {metric}\n\n"
        header += "| shots | " + " | ".join([f"{dataset}" for dataset in filtered_datasets]) + " |\n"
        separator = "|-------|" + "------------|" * (len(filtered_datasets)) + "\n"
        text += header + separator
        

        # Construct the table rows
        for shot_num in range(max_shots + 1):
            row = []
            for dataset in filtered_columns:
                scores = results[shot_num][dataset]
                if scores:
                    mean_score = np.mean(scores)
                    std_score = np.std(scores, ddof=1)
                    row.append(f"{mean_score:.4f} ({std_score:.4f})")
                else:
                    row.append("n/a")
            text += f"|   {shot_num}   | " + " | ".join(row) + " |\n"
        
        text += "\n\n"

    return text
    


def read_result(in_dir, out_file, args): #, metrics=["perplexity"], subset_id=None):
    shots = args.shots
    metrics=args.metrics
    eval_methods=args.eval_methods
    seeds = args.seeds
    subset_id=args.subset_id
    if subset_id is not None:
        if subset_id < 0:
            subset_id = None
    
    # Initialize a set to store unique datasets
    datasets = set()
    
    # Iterate over the directories to identify datasets
    ## Directories take the form of `dataset_shots=3_seed=0`
    for dirname in os.listdir(in_dir):
        if "shots=" in dirname and "seed=" in dirname:
            dataset = dirname.split("_shots=")[0]
            datasets.add(dataset)
    
    # Sort datasets to maintain a consistent order
    datasets = sorted(datasets)

    for eval_method in eval_methods:
        text = per_eval_method_read_result(metrics, datasets, os.path.join(in_dir), eval_method, subset_id=subset_id, shots=shots, seeds=seeds)


        output_file = f"{out_file}-{eval_method}" + ".md"
        out_dir = os.path.dirname(output_file)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        print(eval_method)    
        print(text)
        if text != "":
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)
    
   

def main():
    parser = ArgumentParser()
    parser.add_argument("--in_dir", type=str)
    parser.add_argument("--subset_id", type=int, default=None, help="The dataset to test can be split into `n_subsets`(see run_1d_pdecontrol_eval) and if `subset_id != None` and non-negative, the `subset_id` will be the only subset to find metrics.")
    parser.add_argument("--metrics", type=str, nargs="+", default=[
        "robustness accuracy",
        "robustness mre",
        "robustness failure rate",
        "robustness timeout rate",
        "simulation time mre",
        "edit distance",
        "iou",
        "iou failures",
        "iou timeout rate",
        "perplexity",
        "perplexity timeout rate",
        "gt positive robustness rate",
        "gt negative robustness rate",
        "gt failed robustness rate",
        "adjusted_failure_rate",
        ])
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument(
        "--eval_methods",
        type=str,
        choices=["to_python_direct_with_sstl_cot", "to_python_no_STL", "to_python_two_step", "to_STL"],
        default=["to_python_direct_with_sstl_cot", "to_python_no_STL", "to_python_two_step", "to_STL"],
        nargs='+'
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[-1], help="The seeds to to average results over. A directory may contain multiple seeds. This argument will specify which seeds to average over and the rest will be ignored.")


    
    args = parser.parse_args()
    in_dir = args.in_dir
    out_file = os.path.join(args.in_dir, "results", os.path.basename(in_dir))
    read_result(in_dir, out_file, args=args)
    
if __name__ == "__main__":
    main()
