import argparse
import torch
import os
import logging
from transformers import AutoModelForCausalLM
from peft import PeftModel
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

def main(args):
    model_path = args.model_path
    output_dir = args.output_dir
    adapter_path = args.adapter_path

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f'Loading base model: {model_path}')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
    use_safetensors=True,
    trust_remote_code=True,
    device_map="auto",)

    if hasattr(base_model.config, 'torch_dtype'):
        dtype = base_model.config.torch_dtype
        print("Will save merged model following the base model dtype: ", dtype)
    else:
        dtype =  torch.float32
        print("Couldn't find the base model dtype. Will save merged model in:", dtype)
    
    print(f"loading adapter model: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.to(device)
    model = model.to(dtype)
    print(model)
    model.eval()

    print('Saving merged model')
    merged_model = model.merge_and_unload()
    merged_model_dir = os.path.join(output_dir, "merged_model")
    merged_model.save_pretrained(merged_model_dir, safe_serialization=True)

    print('Saving tokenizer')
    tokenizer.save_pretrained(merged_model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge Model')
    parser.add_argument('--model_path', type=str, help='Path to the model')
    parser.add_argument('--adapter_path', type=str, help='Path to the adapter')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument("--gpus", type=str, default=None, help="Use to set the CUDA_VISIBLE_DEVICES environment variable.")
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        logging.exception(e)
        exit(-1)
