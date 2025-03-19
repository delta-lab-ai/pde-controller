import os
import json
from .few_shot_prompting import FewShotPrompting


class FewShotTrain(FewShotPrompting):
    def __init__(self, num_shots=0, format=None, dataset=None) -> None:
        super().__init__(num_shots=num_shots)
        self.examples = []

        if num_shots > 0:
            # this is only for few-shot training evaluation to get the predicted sstl to construct the second training set of the 2 part training process.
            assert format is not None, "Please provide format for few-shot training evaluation"
            assert dataset is not None, "Please provide dataset for few-shot training evaluation"
            self.format=format
            self.shuffle = True
            current_dir = os.path.dirname(os.path.abspath(__file__))
            jsonl_file_path = os.path.join(current_dir, "examples", f"one_d_{dataset}", "examples.jsonl")
            self.examples = self.load_examples(jsonl_file_path, format)

    def format_prompt(self, format, nl, sstl="", python=""):
        intruction = self.get_intruction(format)
        nl = nl.strip()
        sstl = sstl.strip()
        python = python.strip()
        if format == "nl_to_python":
            prompt = self.get_alpaca_format(intruction, task_input=nl, task_output=python, wrap_in_code_block='python')
            return prompt
        elif format == "nl_to_sstl":
            prompt = self.get_alpaca_format(intruction, task_input=nl, task_output=sstl, wrap_in_code_block='latex')
            return prompt
        elif format == "train_nl_and_sstl_to_python" or format == "train_nl_with_given_sstl_to_python":
            task_input = f"""{nl}\n\nSpatial Signal Temporal Logic:\n```latex\n{sstl}\n```"""
            prompt = self.get_alpaca_format(intruction, task_input=task_input, task_output=python, wrap_in_code_block='python')
            return prompt

    def format_prompt_test(self, nl, sstl="", python=""):
        return self.format_prompt(self.format, nl.strip(), sstl.strip(), python.strip())

    def load_examples(self, jsonl_file_path, format):
        examples = []
        with open(jsonl_file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                nl = data['nl']
                python = data['python']
                sstl = data['sstl']
                example = self.format_prompt(format, nl=nl.strip(), sstl=sstl.strip(), python=python.strip())
                examples.append(example)
        return examples

    def stop_words(self):
        return ["\n### Instruction:", "### Instruction:"]