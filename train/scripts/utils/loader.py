#!/usr/bin/env python3

import copy
import re
import torch
import logging
import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, '../../../utils'))

from few_shot_prompts import FewShotTrain, FewShotDPO

logger = logging.getLogger()

IGNORE_INDEX = -100

class Processor:

    def group_texts(self, examples, tokenizer, max_len):
        input_ids, labels = [], []
        final_input_ids, final_labels = [], []
        
        for idx in range(len(examples['input_ids'])):
            _input_ids = examples['input_ids'][idx]
            _labels = examples['input_ids'][idx]
            examples['input_ids'][idx] = None
            if len(_input_ids) > max_len:
                devided_input_ids, devided_labels = [], []
                for i in range(0, len(_input_ids), max_len):
                    devided_input_ids = _input_ids[i: i + max_len]
                    devided_labels =  _labels[i: i + max_len]
                    if len(devided_input_ids) < max_len:
                        devided_pad_num = max_len - len(devided_input_ids)
                        devided_input_ids += [tokenizer.pad_token_id] * devided_pad_num
                        devided_labels += [IGNORE_INDEX] * devided_pad_num
                    final_input_ids.append(devided_input_ids)
                    final_labels.append(devided_labels)
                continue
                    
            if len(input_ids) + len(_input_ids) > max_len:
                pad_num = max_len - len(input_ids)
                final_input_ids.append(input_ids + [tokenizer.pad_token_id] * pad_num)
                final_labels.append(labels + [IGNORE_INDEX] * pad_num)

                input_ids, labels = [], []
                
            input_ids.extend(_input_ids)
            labels.extend(_labels)
        
        if len(input_ids) > 0:
            pad_num = max_len - len(input_ids)
            final_input_ids.append(input_ids + [tokenizer.pad_token_id] * pad_num)
            final_labels.append(labels + [IGNORE_INDEX] * pad_num)

        return {
            "input_ids": torch.tensor(final_input_ids).long(),
            "labels": torch.tensor(final_labels).long()
        }
    
    
    def truncate_dpo(self, examples, max_len, verbose=False):
        final_prompt_input_ids, final_chosen_input_ids, final_rejected_input_ids = [], [], []
        for idx in range(len(examples['prompt_input_ids'])):
            _prompt_input_ids = examples['prompt_input_ids'][idx]
            _chosen_input_ids = examples['chosen_input_ids'][idx]
            _rejected_input_ids = examples['rejected_input_ids'][idx]

            if len(_prompt_input_ids) > max_len:
                if verbose: print("WARNING: large input longer than context length")
                _prompt_input_ids = _prompt_input_ids[:max_len]
    
    def truncate(self, examples, max_len, verbose=False, dpo=False):
        if dpo:
            return self.truncate_dpo(examples, max_len, verbose=True)
        final_input_ids, final_labels = [], []
        for idx in range(len(examples['input_ids'])):
            _input_ids = examples['input_ids'][idx]
            if 'labels' in examples:
                _labels = examples['labels'][idx]
                if max(len(_input_ids), len(_labels)) > max_len:
                    if verbose: print("WARNING: large input longer than context length")
                    _labels = _labels[:max_len]  # Truncate if necessary
                final_labels.append(_labels)

            if len(_input_ids) > max_len:
                if verbose: print("WARNING: large input longer than context length")
                _input_ids = _input_ids[:max_len]  # Truncate if necessary
            final_input_ids.append(_input_ids)

        if 'labels' in examples:
            return {
                "input_ids": final_input_ids,
                "labels": final_labels,
            }
        return {
            "input_ids": final_input_ids,
        }

    def truncate_and_add_padding(self, examples, tokenizer, max_len, verbose=False):
        final_input_ids, final_labels = [], []
        for idx in range(len(examples['input_ids'])):
            _input_ids = examples['input_ids'][idx]
            if 'labels' in examples:
                _labels = examples['labels'][idx]
                if len(_labels) > max_len:
                    if verbose: print("WARNING: large input labels longer than context length. Truncating...")
                    _labels = _labels[:max_len]  # Truncate if necessary
                final_labels.append(_labels + [IGNORE_INDEX] * (max_len - len(_labels)))
            
            if len(_input_ids) > max_len:
                if verbose: print("WARNING: large input text longer than context length. Truncating...")
                _input_ids = _input_ids[:max_len]  # Truncate if necessary
            final_input_ids.append(_input_ids + [tokenizer.pad_token_id] * (max_len - len(_input_ids)))
        
        if 'labels' in examples:
            return {
                "input_ids": torch.tensor(final_input_ids).long(),
                "labels": torch.tensor(final_labels).long()
            }
        return {
            "input_ids": torch.tensor(final_input_ids).long()
        }
      
    def process_tokenize(self, examples, tokenizer):
        """
        tokenize samples and add bos and eos tokens
        """
        inputs = tokenizer(examples['text'], truncation=False, padding=False)

        input_ids, labels = [], []
        for input_id in inputs['input_ids']:
            if input_id[0] != tokenizer.bos_token_id:
                input_id = [tokenizer.bos_token_id] + input_id
            if input_id[-1] != tokenizer.eos_token_id:
                input_id = input_id + [tokenizer.eos_token_id]
            input_ids.append(input_id)
        
        return {
            "input_ids": input_ids,
        }

    def process_tokenize_sft(self, examples, tokenizer, padding, truncate):
        """
        tokenize samples and add ignore, bos, and eos tokens
        """
        items = [s + t for s, t in zip(examples['text'], examples['labels'])]
        examples_tokenized, sources_tokenized = [tokenizer(strings, truncation=truncate, padding=padding, add_special_tokens=False, max_length=tokenizer.model_max_length) for strings in (items, examples['text'])]

        extracted_sources_tokenized = sources_tokenized['input_ids']
    
        source_lens = [sum(1 for token in input_ids if token != tokenizer.pad_token_id) for input_ids in extracted_sources_tokenized]

        input_ids = []
        for input_id in examples_tokenized["input_ids"]:
            if input_id[0] != tokenizer.bos_token_id:
                input_id = [tokenizer.bos_token_id] + input_id
            if input_id[-1] != tokenizer.eos_token_id:
                input_id = input_id + [tokenizer.eos_token_id]
            input_ids.append(input_id)

        labels = copy.deepcopy(input_ids)

        for label, source_len in zip(labels, source_lens):
            # plus 1 for the bos token
            label[:source_len + 1] = [IGNORE_INDEX] * (source_len + 1)

        return {
            "input_ids": input_ids,
            "labels": labels
        }
    
    def process_tokenize_seq_to_seq(self, examples, tokenizer, truncate=False, padding=False):
        """
        tokenize samples and add bos and eos tokens
        """
        inputs = tokenizer(examples['text'], text_target=examples['labels'], truncation=truncate, padding=padding)

        input_ids, label_ids = [], []
        for input_id, label in zip(inputs['input_ids'], inputs['labels']):
            if input_id[0] != tokenizer.bos_token_id:
                input_id = [tokenizer.bos_token_id] + input_id
            if label[0] != tokenizer.bos_token_id:
                label = [tokenizer.bos_token_id] + label
            
            if input_id[-1] != tokenizer.eos_token_id:
                input_id = input_id + [tokenizer.eos_token_id]
            if label[-1] != tokenizer.eos_token_id:
                label = label + [tokenizer.eos_token_id]

            input_ids.append(input_id)
            label_ids.append(label)
        
        return {
            "input_ids": input_ids,
            "labels": label_ids
        }

    
    def process_tokenize_dpo(self, examples, tokenizer):
        """
        tokenize samples and add bos and eos tokens
        """
        prompt_input_ids = tokenizer(examples["prompt"], add_special_tokens=False)["input_ids"]
        chosen_input_ids = tokenizer(examples["chosen"], add_special_tokens=False)["input_ids"]
        rejected_input_ids = tokenizer(examples["rejected"], add_special_tokens=False)["input_ids"]


        prompts, chosens, rejecteds = [], [], []
        for prompt, chosen, rejected in zip(prompt_input_ids, chosen_input_ids, rejected_input_ids):
            prompts.append(prompt)
            if chosen[-1] != tokenizer.eos_token_id:
                chosen = chosen + [tokenizer.eos_token_id]
            chosens.append(chosen)
            if rejected[-1] != tokenizer.eos_token_id:
                rejected = rejected + [tokenizer.eos_token_id]
            rejecteds.append(rejected)
        
        return {
            "prompt_input_ids": prompts,
            "chosen_input_ids": chosens,
            "rejected_input_ids": rejecteds,
        }

    def create_prompt(self, examples, prompt_format, dataset_class):
        prompter = FewShotTrain()
        texts = []
        labels = []
        if prompt_format == "to_python_no_STL":
            for python, nl in zip(examples['python'], examples['nl']):
                example = prompter.format_prompt('nl_to_python', nl=nl)
                texts.append(example)
                labels.append(python.strip() + "\n```")
        elif prompt_format == "to_STL":
            for nl, sstl in zip(examples['nl'], examples['sstl']):
                example = prompter.format_prompt('nl_to_sstl', nl=nl)
                texts.append(example)
                labels.append(sstl + "\n```")
        elif prompt_format == "to_python_GT_STL":
            for nl, sstl, python in zip(examples['nl'], examples['sstl'], examples['python']):
                example = prompter.format_prompt('train_nl_and_sstl_to_python', nl=nl, sstl=sstl)
                texts.append(example)
                labels.append(python.strip() + "\n```")
        elif prompt_format == "to_python_given_STL":
            for nl, sstl, python in zip(examples['nl'], examples['train_predicted_sstl'], examples['python']):
                example = prompter.format_prompt('train_nl_with_given_sstl_to_python', nl=nl, sstl=sstl)
                texts.append(example)
                labels.append(python.strip() + "\n```")
        elif prompt_format == "to_python_misaligned":
            prompter = FewShotDPO()
            for nl, sstl, python in zip(examples['nl'], examples['stl'], examples['python']):
                example = prompter.format_prompt('dpo_test_sstl_to_python', nl=nl, sstl=sstl)
                texts.append(example)
                labels.append(python.strip() + "\n```")
        else:
            raise ValueError(f"prompter: {prompter} not recognized")
        
        return {
            "text": texts,
            "labels": labels
            }
    
    def create_prompt_dpo(self, examples, prompt_format, dataset_class=None):
        """ Expects keys:  ['anchor', 'w_utility', 'w_time', 'w_sstl', 'l_utility', 'l_time', 'l_sstl', 'dataset_class', 'pidx', 'nc']"""
        prompter = FewShotDPO()
        if prompt_format == "DPO":
            prompts = []
            chosen = []
            rejected = []
            for nl, sstl_w, sstl_l in zip(examples['anchor'], examples['w_sstl'], examples['l_sstl']):
                example = prompter.format_prompt('dpo_train_nl_to_sstl', nl=nl)
                prompts.append(example)
                chosen.append(sstl_w.strip() + "\n```")
                rejected.append(sstl_l.strip() + "\n```")
            return {
                "prompt": prompts,
                "chosen": chosen,
                "rejected": rejected
            }
        