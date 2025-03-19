import json
import os
from .few_shot_test import FewShotTest

class CoTOneDWave(FewShotTest):
    def __init__(self, num_shots, format, dataset="wave"):
        if dataset != "wave" and num_shots != 0:
            raise ValueError(f"Dataset {dataset} not supported. Only 'wave' dataset is supported")
        super().__init__(num_shots)
        self.format = format
        current_dir = os.path.dirname(os.path.abspath(__file__))
        jsonl_file_path = os.path.join(current_dir, "examples", "one_d_wave", "examples.jsonl")
        self.examples = self.load_examples(jsonl_file_path, format)

    def load_examples(self, jsonl_file_path, format):
        examples = []
        with open(jsonl_file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                nl = data['nl']
                python = data['python']
                sstl = data['sstl']
                example = super().format_prompt(format, nl=nl.strip(), sstl=sstl.strip(), python=python.strip())
                examples.append(example)
        return examples

    def format_prompt(self, nl="", sstl="", python=""):
        few_shots = self._get_few_shot_prompt()
        curr_prompt = super().format_prompt(self.format, nl, sstl, python)
        return f"{few_shots}{curr_prompt}"