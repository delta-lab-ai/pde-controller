#!/usr/bin/env python3

import inspect
import os
import random
import shutil
import logging
import transformers
import torch

import torch.distributed as dist

from datetime import datetime
from dataclasses import field, dataclass
from utils.util import set_logger, print_args

from utils.loader import Processor
from utils.trainer import LoggerCallback, RemoveStateCallback, StepCheckpointCallback, DataCollatorForDPODataset

from datasets import load_dataset
from transformers import (
    set_seed,
    AutoConfig,
    AutoTokenizer, 
    HfArgumentParser,
    TrainingArguments, 
    AutoModelForCausalLM,
)

from peft import LoraConfig, PeftModel
from trl import DPOTrainer, DPOConfig

logger = logging.getLogger()

@dataclass
class DataArguments:

    no_timestamps: bool = field(default=False)
    
    
    # data
    train_parquet_file: str = field(default=None)
    train_file_config: str = field(default=None)
    train_dataset: str = field(default=None)
    train_coef: str = field(default=None)
    delete_long_sample: bool = field(default=False)

    # process
    max_len: int = field(default=4096)
    preprocessing_num_workers: int = field(default=64)
    
    # model
    model_cfg: str = field(default="data/models/starcoder")
    adapter_cfg: str = field(default="data/models/starcoder")
    flash_attention: bool = field(default=False)

    # LoRA model
    save_merged_model: bool = field(default=False)
    no_load_model_pararmeters: bool = field(default=False)
    resume_from: str = field(default=None)

    resume_step: int = field(default=None)
    resume_batch_size: int = field(default=None)

    # output
    stream: bool = field(default=False)

    step_save_interval: int = field(default=1000) # save model every n steps. These will persist.

    external_validation: bool = field(default=False)

@dataclass
class PeftArguments:
    ## See https://github.com/huggingface/peft/blob/f0b066eae888d5dea598e756b7e6d3401d0708e7/src/peft/tuners/lora/config.py#L72
    ##   for the default values of the fields (or define more or new defaults here).
    # some_field: str = field(default="default_value")
    target_modules: str = field(default="k_proj,down_proj,q_proj,v_proj,gate_proj,o_proj,up_proj")
    task_type: str = field(default="CAUSAL_LM")

    lora_r: int = field(default=16)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.1)
    bias: str = field(default="none")

def train():
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    print("Rank", rank, "Current device", torch.cuda.current_device())

    parser = HfArgumentParser((DataArguments, TrainingArguments, PeftArguments))

    data_args, training_args, peft_args = parser.parse_args_into_dataclasses()

    training_args._frozen = False

    if not data_args.no_timestamps:
        timestr = datetime.now().strftime("-%m%d%H%M")
        training_args.output_dir = training_args.output_dir + timestr

    training_args.logging_dir = os.path.join(training_args.output_dir, 'logging')

    if os.path.exists(training_args.output_dir):
        if training_args.overwrite_output_dir:
            print(f"Output directory ({training_args.output_dir}) already exists. Overwriting output dir.")
            if training_args.process_index == 0:
                shutil.rmtree(training_args.output_dir)
        else:
            print(f"Output directory ({training_args.output_dir}) already exists. Use --overwrite_output_dir to overcome.")
    
    if training_args.world_size > 1:
        dist.barrier(device_ids=[rank])
    
    if training_args.process_index == 0:
        if not os.path.exists(training_args.output_dir):
            os.makedirs(training_args.output_dir)
    
    if training_args.world_size > 1:
        dist.barrier(device_ids=[rank])
    
    set_seed(training_args.seed)

    node_rank = int(os.getenv('GROUP_RANK', '0'))

    for _logger in [logger, transformers.utils.logging.get_logger(), logging.getLogger('DeepSpeed')]:
        set_logger(_logger, training_args.local_rank, data_args.stream, os.path.join(training_args.output_dir, f'log-node-{node_rank}.log'))

    logger.warning("Device: %s, rank: %s, world size: %s", training_args.device, training_args.process_index, training_args.world_size)

    if training_args.world_size > 1:
        dist.barrier(device_ids=[rank])

    print_args(data_args, 'Data Arguments')
    print_args(training_args, 'Training Arguments')
    print_args(peft_args, 'LoRA Arguments')

    config = AutoConfig.from_pretrained(data_args.model_cfg, trust_remote_code=True)
    config._attn_implementation = "flash_attention_2"
    config.use_cache = False

    # load base model
    base_model = AutoModelForCausalLM.from_pretrained(data_args.model_cfg, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True,)
    logger.info(base_model)
    base_model.config.use_cache = False
    ## load the same adapter twice. once in training mode and once in eval mode.
    model = PeftModel.from_pretrained(
        base_model,
        data_args.adapter_cfg,
        is_trainable=True,
        adapter_name='policy',
        )
    # load the adapter a second time, with a different name, which will be our reference model.
    model.load_adapter(
        data_args.adapter_cfg,
        adapter_name="reference",
        )

    tokenizer = AutoTokenizer.from_pretrained(data_args.model_cfg, trust_remote_code=True)
    tokenizer.padding_side = 'right'
    logger.info(f"padding side: {tokenizer.padding_side}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        
    print("Loading training data", data_args.train_parquet_file)
    if data_args.train_parquet_file is not None:
        train_sets = load_dataset("parquet", data_files=data_args.train_parquet_file, split='train')
    else:
        raise ValueError("Should provide either 'train_dataset' or 'train_file_config'")
    
    logger.info('Total %d case', len(train_sets))
    
    with training_args.main_process_first(desc="Log a few random samples from the training set"):
        for index in random.sample(range(len(train_sets)), 3):
            ### DPOTrainer implements the tokenizer itself. no way to turn it off short of editing the source code.
            logger.info(
                "Sample %d of the raw training set:\n\nprompt_input_ids: %s\n\nprompt_tokens: %s\n\nchosen_input_ids: %s\n\nchosen_tokens: %s\n\nrejected_input_ids: %s\n\nrejected_tokens: %s\n\n",
                index, 
                train_sets[index]['prompt_input_ids'],
                tokenizer.convert_ids_to_tokens(train_sets[index]['prompt_input_ids']),
                train_sets[index]['chosen_input_ids'],
                tokenizer.convert_ids_to_tokens(train_sets[index]['chosen_input_ids']),
                train_sets[index]['rejected_input_ids'],
                tokenizer.convert_ids_to_tokens(train_sets[index]['rejected_input_ids']),
            )
    
    train_sets = train_sets.shuffle(seed=training_args.seed)            

    epoch_checkpoint_callback = StepCheckpointCallback(save_interval=data_args.step_save_interval, output_dir=training_args.output_dir, external_validation=data_args.external_validation)

    data_collator = DataCollatorForDPODataset(tokenizer=tokenizer)

    target_modules_str = peft_args.target_modules
    # convert the comma-separated string to a list of strings
    if target_modules_str:
        target_modules_list = [module.strip() for module in target_modules_str.split(',')]
    else:
        target_modules_list = None

    lora_config = LoraConfig(
        target_modules=target_modules_list,
        task_type=peft_args.task_type,
        r=peft_args.lora_r,
        lora_alpha=peft_args.lora_alpha,
        lora_dropout=peft_args.lora_dropout,
        bias=peft_args.bias,
    )

    print(lora_config)

    training_args_main_output_dir = training_args.output_dir
    training_args.output_dir = os.path.join(training_args.output_dir, f"backups")


    ### DPO Config initialized this way since even the defaults will override the transformer.TrainingArguments
    # Get the list of parameters that DPOConfig accepts
    dpo_config_params = inspect.signature(DPOConfig).parameters
    # Filter training_args to include only those parameters
    relevant_args = {k: v for k, v in vars(training_args).items() if k in dpo_config_params}
    # Add any additional parameters that are not in training_args
    relevant_args.update({
        'ref_adapter_name': "reference",
        'model_adapter_name': "policy",
        'beta': 0.1,
        'loss_type': "sigmoid",
        'is_encoder_decoder': False, 
        'rpo_alpha': 1.0, 
        'sync_ref_model': False, 
        'ref_model_sync_steps': 4, # only used if the sync_ref_module is set to True
        'force_use_ref_model': False,
    })


    dpo_config = DPOConfig(**relevant_args)
    logger.info(dpo_config)

    dpo_trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_sets,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LoggerCallback, epoch_checkpoint_callback],
    )

    epoch_checkpoint_callback.set_trainer(dpo_trainer)

    dpo_trainer.train(resume_from_checkpoint=data_args.resume_from)
    
    dpo_trainer.save_model(os.path.join(training_args_main_output_dir, "final"))

if __name__ == "__main__":

    try:
        train()
    except Exception as e:
        logging.exception(e)
        exit(-1)
