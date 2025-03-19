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
    # Put into file_paths
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
        return None
        # raise ValueError(f"Dataset {keywords} does not contain one of 'heat' or 'wave'.")


def name_output_file(in_file, out_dir):
    """Create a name for the output file using keywords in the base-file name and base-directory name.
    Does not include the .parquet file extension."""
    parent_folder = os.path.basename(os.path.dirname(in_file))
    basename_without_extension = os.path.splitext(os.path.basename(in_file))[0]
    # Split into words
    parent_words = parent_folder.split('_')
    basename_words = basename_without_extension.split('_')
    # Combine and remove duplicates
    unique_keywords = '_'.join(list(dict.fromkeys(parent_words + basename_words)))
    dataset_class = get_dataset_class(unique_keywords)
    out_file = os.path.join(out_dir, "tokenized_" + "".join(unique_keywords) + ".parquet")
    return str(out_file), dataset_class



def prosess_tokenize(in_file, out_dir, tokenizer, prompt_format, truncate=False, padding=False):
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
        processor.create_prompt_dpo,
        fn_kwargs={
            "prompt_format": prompt_format
        },
        batched=True,
        load_from_cache_file=False,
        batch_size=process_batch_size,
        num_proc=64,
        desc=f"Creating prompt in format: {prompt_format}, in the standard data'{{'prompt', 'chosen', 'rejected'}}'.",
        )

    columns_to_remove = [feature for feature in _dataset.features if feature not in ["prompt", "chosen", "rejected"]]
    _dataset = _dataset.remove_columns(columns_to_remove)
    
    print("Example from the processed dataset:")
    random_indices = random.sample(range(len(_dataset)), 2)
    for i in random_indices:
        print("\nexample", i, ":")
        print(_dataset[i])


    column_names = list(_dataset.features)
    ## dpo
    # DPOTrainer will do the tokenization of the prompt, chosen, and rejected in the training loop.
    _dataset = _dataset.map(
        processor.process_tokenize_dpo,
        fn_kwargs={
            "tokenizer": tokenizer,
        },
        batched=True,
        load_from_cache_file=False,
        remove_columns=column_names,
        batch_size=process_batch_size,
        num_proc=64,
        desc="Running tokenizer on DPO dataset",
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

    # Put file_paths out for debugging
    current_dir_path = os.path.dirname(os.path.realpath(__file__))
    out_file = os.path.join(current_dir_path, f"file-names_to-be-tokenized.json")
    with open(out_file, "w") as f:
        json.dump(file_paths, f)

    return file_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="+", help="List of paths directories containing json files of training data that will be preprocessed into individual parquet files for training.")
    parser.add_argument("--out_file_path", type=str, default=None, help="Path to directory where to store the output parquet file") 
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B", help="Path to the model that will be trained and whose tokenizer will be use here to preprocess the data.")
    parser.add_argument("--prompt_format", choices=["DPO"], default="DPO", help="""Choose the prompt format, the Dataset requires corresponding keys.
                        `DPO`: No STL is given in the prompt. The model is given natural language and should directly convert to STL. Required keys: 'nl', 'sstl'.
                        """)
    ## Padding and truncating is removed from this code because it should be done in the grouping stage (which can also combine other datasets so they all have the same context length).
    # parser.add_argument("--truncate", action="store_true", help="Whether to truncate the input and labels in tokenizer if either exceeds the context length of the model. If you plan to group the training data together then do not truncate here. Truncation will happen during grouping.")
    # parser.add_argument("--padding", action="store_true", help="Whether to pad the input and labels in tokenizer if either is shorter than the context length of the model. If you plan to group the training data together then do not pad here. Padding will happen during grouping. If you don't plan to group the data, padding here is equivalent to padding during grouping stage (just skipping the grouping part).")

    args = parser.parse_args()

    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    file_paths = get_all_files_in_directories(args.paths)

    out_dir = args.out_file_path
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    paths_to_be_trained = []
    for in_file in file_paths:
        path = prosess_tokenize(in_file, out_dir, tokenizer, args.prompt_format)
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