import argparse
import os
import sys
import traceback
current_dir = os.path.dirname(os.path.realpath(__file__))
from tqdm import tqdm
import json
from pebble import ProcessPool
from concurrent.futures import TimeoutError
import random



utils_path = os.path.join(current_dir, '../../../../utils')
sys.path.append(utils_path)

from few_shot_prompts import CoTOneDHeat, CoTOneDWave


evaluation_path = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(evaluation_path)

eval_path = os.path.abspath(os.path.join(current_dir, '../eval'))
sys.path.append(eval_path)

data_processing_path = os.path.abspath(os.path.join(current_dir, '../data_processing'))
sys.path.append(data_processing_path)

from data_processing.answer_extraction import *
from eval.eval_script import *




# model_name_or_path cannot be both None or both not None.
model = None
tokenizer = None
pool = None
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def evaluate(eval_fn, tasks, _timeout=15):
    with ProcessPool() as pool:
        timeout_cnt = 0
        failure_cnt = 0
        iterator = pool.map(eval_fn, tasks, timeout=_timeout).result()
        labels = []
        while True:
            try:
                labels.append((next(iterator)))
            except StopIteration:
                break
            except TimeoutError as error:
                labels.append(0)
                timeout_cnt += 1
            except Exception as error:
                print("An error occurred:", error, flush=True)
                traceback.print_exception(type(error), error, error.__traceback__, file=sys.stdout)
                failure_cnt += 1
                labels.append(-100)
    return labels, timeout_cnt, failure_cnt

def evaluate_future(eval_fn, tasks, _timeout=300):
    # The runtime is doubled because the gt may have to be simulated also.
    num_cpus = os.cpu_count()
    max_workers = max(1, int(np.floor(num_cpus * 0.5)))
    print(f"Using {max_workers} out of {num_cpus} CPUs")
    with ProcessPool(max_workers=max_workers) as pool:
        timeout_cnt = 0
        futures = [(i, pool.schedule(eval_fn, args=(task,), timeout=_timeout)) for i, task in enumerate(tasks)]
        results = [None] * len(tasks)

        with tqdm(total=len(futures), desc="Evaluating robustness (completed)") as pbar:
            for i, future in futures:
                try:
                    result = future.result() 
                    gt_robustness, gt_simtime = result
                    results[i] = (gt_robustness, gt_simtime)

                except TimeoutError:
                    results[i] = ("timeout", "timeout")
                    timeout_cnt += 1
                except Exception as error:
                    print("An error occurred:", error, flush=True)
                    traceback.print_exception(type(error), error, error.__traceback__, file=sys.stdout)
                    exit()
                finally:
                    pbar.update(1)
    list_gt_robustness, list_gt_simtime = zip(*results)
    return list_gt_robustness, list_gt_simtime, timeout_cnt


def main(args):
    if args.gpus is not None: 
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    random.seed(42)

    print("Loading data...")
    test_data = []
    with open(os.path.join(args.data_dir, f"validation.jsonl" if args.infer_on_train_set else f"test.jsonl")) as fin:
        for line in fin:
            example = json.loads(line)
            python_code = example[args.python_key]
            sstl = example[args.stl_key]
            natural_language = example[args.nl_key]
            robustness = example.get(args.robustness_key, None)
            example['python'] = python_code.strip()
            example['sstl'] = sstl
            example['nl'] = natural_language
            if robustness is not None:
                example['robustness'] = robustness
            test_data.append(example)

    if args.max_num_examples and len(test_data) > args.max_num_examples:
        test_data = random.sample(test_data, args.max_num_examples)

    if not test_data:
        print("Ending. There was no data to test.")
        return

    args.save_dir = os.path.join(args.data_dir + "_" + str(args.max_num_examples))
    os.makedirs(args.save_dir, exist_ok=True)


    if args.eval_robustness:
        eval_robustness_function = eval("eval_robustness_gt")
        sim_gt_robustnesses, sim_gt_simulation_times, eval_timeout_cnt_robustness = evaluate_future(eval_robustness_function, test_data)
        print("done evaluating robustness", flush=True)
        for item, gt_r, gt_time in zip(test_data, sim_gt_robustnesses, sim_gt_simulation_times):
            if 'robustness' not in item:
                item['robustness'] = gt_r
                item['time'] = gt_time

    print("Calculating accuracy...")
    ## track dataset stats:
    sum_gt_positive_robustness = 0
    sum_gt_negative_robustness = 0
    sum_gt_failed_robustness = 0
    for item in test_data:
        if args.eval_robustness:
            ## track dataset stats:
            gt_r = item['robustness']
            if gt_r > 0:
                sum_gt_positive_robustness += 1
            elif gt_r < 0 and gt_r != -100:
                sum_gt_negative_robustness += 1
            elif gt_r == -100:
                sum_gt_failed_robustness += 1
            else:
                raise ValueError(f"gt_r = {gt_r}")
            ## end track dataset stats
            
    
    if args.eval_robustness: 
        ## track dataset stats:
        gt_positive_robustness_rate = sum_gt_positive_robustness / len(test_data)
        gt_negative_robustness_rate = sum_gt_negative_robustness / len(test_data)
        gt_failed_robustness_rate = sum_gt_failed_robustness / len(test_data)
        print(f"gt positive robustness rate = {gt_positive_robustness_rate * 100}", flush=True)
        print(f"gt negative robustness rate = {gt_negative_robustness_rate * 100}", flush=True)
        print(f"gt failed robustness rate = {gt_failed_robustness_rate * 100}", flush=True)


    if args.eval_robustness: print(f"Timeout count >>> output eval robustness = {eval_timeout_cnt_robustness}", flush=True)

    with open(os.path.join(args.save_dir, f"validation.jsonl" if args.infer_on_train_set else f"test.jsonl"), "w") as fout:
        for item in test_data:
            if 'sstl' in item:
                sstl = item.pop('sstl')
                item['sstl'] = sstl
            if 'python' in item:
                python = item.pop('python')
                item['python'] = python
            if 'robustness' in item:
                robustness = item.pop('robustness')
                item['robustness'] = robustness
            if 'time' in item:
                time = item.pop('time')
                item['time'] = time
            json.dump(item, fout, ensure_ascii=True)
            fout.write("\n")

    metric_fname = "metrics.json"
    with open(os.path.join(args.save_dir, metric_fname), "w") as fout:
        metrics = {
            "n_samples": len(test_data),
        }
        if args.eval_robustness:
            metrics["gt positive robustness rate"] = gt_positive_robustness_rate
            metrics["gt negative robustness rate"] = gt_negative_robustness_rate
            metrics["gt failed robustness rate"] = gt_failed_robustness_rate
        json.dump(metrics, fout, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="random seed for the LLM generation only. This SHOULD NOT affect the random seed for data selection (I did not debug this so set it at your own peril).")
    parser.add_argument("--data_dir", type=str, default="data/mgsm")
    parser.add_argument("--python_key", type=str, default="python", help="Key for accessing the Python code in the example.")
    parser.add_argument("--stl_key", type=str, default="sstl", help="Key for accessing the STL in the example.")
    parser.add_argument("--nl_key", type=str, default="nl", help="Key for accessing the natural language in the example.")
    parser.add_argument("--robustness_key", type=str, default="robustness", help="Key for accessing the robustness in the example.")
    
    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate.")
    parser.add_argument("--save_dir", type=str, default="results/mgsm")
    
    parser.add_argument("--infer_on_train_set", action="store_true")
    
    parser.add_argument("--eval_perplexity", action='store_true', help="Whether to evaluate the perplexity of the model outputs.")
    parser.add_argument("--eval_robustness", action='store_true', help="Whether to evaluate the robustness of the model outputs by comparing on the FemFormal repo.")
    parser.add_argument("--eval_edit_distance", action='store_true', help="Whether to evaluate the edit distance of the model python with the ground truth python.")    
    parser.add_argument("--eval_iou", action='store_true', help="Whether to evaluate the precision and recall of the model sstl latex generation with the ground truth sstl.")    
    parser.add_argument("--load_from_file", action='store_true', help="Whether to load the model predictions from a file instead of regenerating them. If this flag is provided but no file exists, the model predictions will be generated and saved to a file.")
    parser.add_argument("--skip_existing_scores", action='store_true', help="Whether to skip computing and overwriting the evaluation of a datapoint (ex. --eval_perplexity) that exists already. Only relevant if `--load_from_file` is provided.")
    parser.add_argument("--gpus", type=str, default=None, help="Use to set the CUDA_VISIBLE_DEVICES environment variable.")
    args, unparsed_args = parser.parse_known_args()
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    print("unparsed args:", flush=True)
    print(unparsed_args, flush=True)
    print("args:", flush=True)
    print(args, flush=True)

    if args.seed is not None:
        if args.seed < 0:
            args.seed = None

    if 'math6' in args.data_dir:
        args.multi_turn = True

    # the basename of the datadir is the dataset name
    args.dataset_name = args.data_dir.split("/")[-1]

    main(args)

    if pool is not None:
        pool.close()
