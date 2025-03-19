from .few_shot_test import FewShotTest
import json
import os

class CoTOneDCombined(FewShotTest):
    def __init__(self, num_shots, format, dataset="combined"):
        assert dataset in ["combined", "heat", "wave"]
        if dataset== "combined" and num_shots not in [0, 2]:
            raise ValueError(f"Number of shots must be 0 or 2 for dataset {dataset}")
        super().__init__(num_shots)
        self.format=format
        current_dir = os.path.dirname(os.path.abspath(__file__))
        jsonl_file_path = os.path.join(current_dir, "examples", f"one_d_{dataset}", "examples.jsonl")
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