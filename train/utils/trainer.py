#!/usr/bin/env python3
import os
import glob
import logging
import datetime

from transformers import TrainerCallback
from transformers import TrainingArguments, TrainerState, TrainerControl
import os

import torch

from typing import Dict, Sequence
from dataclasses import dataclass
import transformers


IGNORE_INDEX = -100

logger = logging.getLogger()

class LoggerCallback(TrainerCallback):

    def on_train_begin(self, args, state, control, **kwargs):
        
        self.start_time = datetime.datetime.now()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_local_process_zero:
            return
        
        if 'loss' not in logs:
            return
        
        loss_msg = ' '.join(["%s: %s" % (k, v) for k, v in logs.items() if 'loss' in k])
        now = datetime.datetime.now()
        pass_time = now - self.start_time
        rest_time = pass_time * (state.max_steps - state.global_step) / state.global_step
        eta = now + rest_time

        pt_min = pass_time.seconds // 60
        pass_time = '%.2d:%.2d' % (pt_min // 60 + pass_time.days * 24, pt_min % 60)

        rt_min = rest_time.seconds // 60
        rest_time = '%.2d:%.2d' % (rt_min // 60 + rest_time.days * 24, rt_min % 60)

        logger.info(
            'step: %d epoch: %.2f %s lr: %.4g passed time: %s rest time: %s eta: %s',
            state.global_step, state.epoch, loss_msg, logs.get('learning_rate', 0),
            pass_time, rest_time, eta.strftime('%m/%d %H:%M')
        )

class RemoveStateCallback(TrainerCallback):

    def remove_state(self, args, step):
        step = int(step)

        if step <= 0:
            return

        step_dir =  os.path.join(args.output_dir, f'checkpoint-{step}')
        logger.info('Remove state in %s', step_dir)

        remove_paths = [
            os.path.join(step_dir, 'latest'), # deepspeed state
            os.path.join(step_dir, f'global_step{step}'), # deepspeed state
            os.path.join(step_dir, 'optimizer.pt'), # optimizer state
            os.path.join(step_dir, 'scheduler.pt'), # scheduler state
            os.path.join(step_dir, 'generation_config.json'), # generation config
            os.path.join(step_dir, 'trainer_state.json'), # trainer state
            os.path.join(step_dir, 'training_args.bin'), # training args
            os.path.join(step_dir, 'zero_to_fp32.py')
        ]

        remove_paths.extend(glob.glob(os.path.join(step_dir, 'rng_state_*.pth'))) # numpy random state

        for path in remove_paths:
            if os.path.exists(path):
                os.system('rm -rf %s' % path)

    def on_save(self, args, state, control, **kwargs):

        if not state.is_world_process_zero:
            return
        
        self.remove_state(args, state.global_step - state.save_steps)
    
    def on_train_end(self, args, state, control, **kwargs):
        
        if not state.is_world_process_zero:
            return
        
        self.remove_state(args, state.global_step)


class StepCheckpointCallback(TrainerCallback):
    def __init__(self, save_interval, output_dir, external_validation):
        self.save_interval = save_interval
        self.output_dir = output_dir
        self.trainer = None
        self.external_validation = external_validation

    def set_trainer(self, trainer):
        self.trainer = trainer

    def save_model(self, checkpoint_dir, state: TrainerState):
        self.trainer.save_model(checkpoint_dir)
        print(f"Global step: {state.global_step} (Epoch {int(state.epoch)}). Saved checkpoint at {checkpoint_dir}")


    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.save_interval == 0:
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-step-{state.global_step}")
            self.save_model(checkpoint_dir, state)
            control.should_save = True # save to resume later
            if self.external_validation:
                control.should_training_stop = True  # signal to stop training
            return control

        

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning.
    
    Will do right padding for input_ids and labels up to the longest sequence in the batch."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([torch.tensor(instance[key]) for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    



@dataclass 
class DataCollatorForDPODataset(object):
    """Collate examples for DPO fine-tuning.
    
    Will do right padding for input_ids and labels up to the longest sequence in the batch."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, chosen_ids, rejected_ids = tuple([torch.tensor(instance[key]) for instance in instances] for key in ("prompt_input_ids", "chosen_input_ids", "rejected_input_ids"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        chosen_ids = torch.nn.utils.rnn.pad_sequence(
            chosen_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        rejected_ids = torch.nn.utils.rnn.pad_sequence(
            rejected_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        return dict(
            prompt_input_ids=input_ids,
            prompt_attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            chosen_input_ids=chosen_ids,
            chosen_attention_mask=chosen_ids.ne(self.tokenizer.pad_token_id),
            rejected_input_ids=rejected_ids,
            rejected_attention_mask=rejected_ids.ne(self.tokenizer.pad_token_id),
        )
    