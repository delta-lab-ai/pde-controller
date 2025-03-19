import argparse
import json
import os
import sys
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from utils.loader import Processor
import random



os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))


def create_file_names(file_paths, out_file):
    with open(out_file, "w") as f:
        json.dump(file_paths, f)

        
def load_jsonl(in_file):
    with open(in_file, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]
    return datas


def save_jsonl(datas, out_file):
    with open(out_file, "w", encoding="utf-8") as f:
        for data in datas:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def get_dataset_class(keywords):
    """Get the dataset class from the keywords in the provided dataset name. One of "heat", "wave"."""
    if "heat" in keywords:
        return "heat"
    elif "wave" in keywords:
        return "wave"
    else:
        raise ValueError(f"Dataset {keywords} does not contain one of 'heat' or 'wave'.")


def name_output_file(in_file, out_dir):
    """Create a name for the output file using keywords in the base-file name and base-directory name.
    Does not include the .parquet file extension."""
    parent_folder = os.path.basename(os.path.dirname(in_file))
    basename_without_extension = os.path.splitext(os.path.basename(in_file))[0]
    # split into words
    parent_words = parent_folder.split('_')
    basename_words = basename_without_extension.split('_')
    # combine and remove duplicates
    unique_keywords = '_'.join(list(dict.fromkeys(parent_words + basename_words)))
    dataset_class = get_dataset_class(unique_keywords)
    out_file = os.path.join(out_dir, "tokenized_" + "".join(unique_keywords) + ".parquet")
    return str(out_file), dataset_class



def prosess_tokenize(in_file, out_dir, tokenizer, prompt_format, sft=False, truncate=False, padding=False):
    """Tokenize the data in the input file, and save the tokenized data in the output directory as a parquet file.
        Will returned the name of the saved file."""
    out_file, dataset_class = name_output_file(in_file, out_dir)
    if os.path.isfile(out_file):
        print(f"Warning: {out_file} already exists. Will replace")
        os.remove(out_file)

    processor = Processor()
    _dataset = load_dataset(in_file.split(".")[-1] if in_file.split(".")[-1] != "jsonl" else "json", data_files=in_file, split='train')
    process_batch_size = min(2, len(_dataset))

    print(_dataset)
    _dataset = _dataset.map(
        processor.create_prompt,
        fn_kwargs={
            "prompt_format": prompt_format,
            "dataset_class": dataset_class
        },
        batched=True,
        load_from_cache_file=False,
        batch_size=process_batch_size,
        num_proc=64,
        desc=f"Creating prompt in format: {prompt_format}, in the standard data'{{'text', 'labels'}}'.",
        )

    columns_to_remove = [feature for feature in _dataset.features if feature not in ["text", "labels"]]
    _dataset = _dataset.remove_columns(columns_to_remove)
    
    print("Example from the processed dataset:")
    random_indices = random.sample(range(len(_dataset)), 2)
    for i in random_indices:
        print("\nexample", i, ":")
        print(_dataset[i])


    column_names = list(_dataset.features)
    if sft:
        _dataset = _dataset.map(
            processor.process_tokenize_sft,
            fn_kwargs={
                "tokenizer": tokenizer,
                "truncate": truncate,
                "padding": padding
            },
            batched=True,
            load_from_cache_file=False,
            remove_columns=column_names,
            batch_size=process_batch_size,
            num_proc=64,
            desc=f"Running tokenizer on SFT dataset. padding: {padding}, truncation: {truncate}.",
        )
    else:
        _dataset = _dataset.map(
            processor.process_tokenize,
            fn_kwargs={
                "tokenizer": tokenizer,
                "truncate": truncate,
                "padding": padding
            },
            batched=True,
            load_from_cache_file=False,
            remove_columns=column_names,
            batch_size=process_batch_size,
            num_proc=64,
            desc="Running tokenizer on pretraining dataset",
        )

    print(_dataset)
    _dataset.to_parquet(out_file)
    return str(out_file)

def get_all_files_in_directories(directories):
    file_paths = []
    for in_dir in directories:
        dir_list = os.listdir(in_dir)
        for file_name in dir_list:
            full_path = os.path.join(in_dir, file_name)
            # ignore markdown/jsonl description files of datasets
            if os.path.isfile(full_path) and full_path.split(".")[-1] != "md" and "description" not in file_name.split(".")[0]:
                file_paths.append(full_path)

    # send file_paths out for debugging
    current_dir_path = os.path.dirname(os.path.realpath(__file__))
    out_file = os.path.join(current_dir_path, f"file-names_to-be-tokenized.json")
    with open(out_file, "w") as f:
        json.dump(file_paths, f)

    return file_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="+", help="List of paths directories containing json files of training data that will be preprocessed into individual parquet files for training.")
    parser.add_argument("--sft", action="store_true", help="""Tokenize as supervised fine-tuning for text-label data. For SFT, output data will contain:
                        data["input_ids"], data["labels"], where data["labels"] = len(data["input_ids"]) * [-100] + `true labels`. [-100] is the default ignore token for cross-entropy loss.
                        If not set, then the output data will use the pretraining objective: data["input_ids"], data["labels"], where data["labels"] = data["input_ids"].""")
    parser.add_argument("--out_file_path", type=str, default=None, help="Path to directory where to store the output parquet file") 
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B", help="Path to tokenizer.")
    parser.add_argument("--prompt_format", choices=["to_python_no_STL", "to_STL", "to_python_GT_STL", "to_python_given_STL", "to_python_misaligned"], default="to_python_no_STL", help="""Choose the prompt format, the Dataset requires the corresponding keys.
                        `to_python_no_STL`: No STL is given in the prompt. The model is given natural language and should directly convert to code. Required keys: 'nl', 'python'.
                        `to_STL`: No STL is given in the prompt. The model is given natural language and should directly convert to STL. Required keys: 'nl', 'sstl'.
                        `to_python_GT_STL`: Ground truth STL is given in the prompt. The model is given natural language and STL and should directly convert to code. Required keys: 'nl', 'sstl', 'python'.
                        `to_python_given_STL`: Must provide STL to be given in the prompt. The model is given natural language and STL and should directly convert to code. Required keys: 'nl', 'predicted_sstl', 'python'.
                        `to_python_misaligned`: Must provide STL to be given in the prompt. The model is given natural language and STL which do not necessarily describe the same problem and should directly convert to code. Required keys: 'nl', 'predicted_sstl', 'python'.
                        """)

    args = parser.parse_args()

    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    file_paths = get_all_files_in_directories(args.paths)

    out_dir = args.out_file_path
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    paths_to_be_trained = []
    for in_file in file_paths:
        path = prosess_tokenize(in_file, out_dir, tokenizer, args.prompt_format, args.sft)
        if path in paths_to_be_trained:
            raise ValueError(f"{path} was made twice in this script. The fist has already been overwritten in the function `process_tokenize`. Check the dataset names and try again.")
        else:
            paths_to_be_trained.append(path)


    print(os.getcwd())
    dir_path = os.path.dirname(os.path.realpath(__file__))
    out_file = os.path.join(dir_path, "file_names_to-be-trained.json")
    create_file_names(paths_to_be_trained, out_file)
        

if __name__ == "__main__":
    main()