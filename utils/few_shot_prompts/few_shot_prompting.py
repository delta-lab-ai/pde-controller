import copy
import random


class FewShotPrompting:
    def __init__(self, num_shots):
        self.num_shots = num_shots
        if num_shots > 3:
            raise ValueError("Only supports 0 up to 3 shots.")
        pass

    def get_alpaca_format(self, instruction, task_input, task_output="", wrap_in_code_block=None) -> str:
        if wrap_in_code_block == 'python':
            prompt = f"""### Instruction:\n{instruction}\n\n### Input:\n{task_input}\n\n### Response:\n```python\n{task_output}"""
            if task_output != "":
                prompt += "\n```\n\n"
            return prompt
        elif wrap_in_code_block == 'latex':
            prompt = f"""### Instruction:\n{instruction}\n\n### Input:\n{task_input}\n\n### Response:\n```latex\n{task_output}"""
            if task_output != "":
                prompt += "\n```\n\n"
            return prompt
        else:
            if task_output != "":
                return f"""### Instruction:\n{instruction}\n\n### Input:\n{task_input}\n\n### Response:\n{task_output}\n\n"""
            else:
                return f"""### Instruction:\n{instruction}\n\n### Input:\n{task_input}\n\n### Response:\n"""



    def _get_few_shot_prompt(self):
        if hasattr(self, 'examples'):
            if hasattr(self, 'shuffle'):
                if self.shuffle:
                    temp_examples = copy.copy(self.examples)
                    random.shuffle(temp_examples)
                    return "".join(temp_examples[:self.num_shots])
            prompt_examples = self.examples[:self.num_shots]
        else:
            raise AttributeError("Subclasses of FewShotPrompting must define 'self.examples'")
        return "".join(prompt_examples)
    
    def get_intruction(self, format):
        if format == "nl_to_python":
            return """Below is a natural language description of partial differential equation optimization problem. Translate the problem into Python code following spatial-signal temporal logic."""
        elif format == "nl_to_sstl":
            return """Below is a natural language description of partial differential equation optimization problem. Translate the problem into Latex code following spatial-signal temporal logic."""
        elif format == "train_nl_and_sstl_to_python":
            return """Below is a natural language description of partial differential equation optimization problem, paired with a spatial-signal temporal logic description of the same problem. Translate the problem into Python code following spatial-signal temporal logic."""
        elif format == "test_nl_to_python_with_sstl_cot":
            return """Below is a natural language description of partial differential equation optimization problem. Translate the problem into Python code following spatial-signal temporal logic. Explain your reasoning by first providing spatial signal temporal logic statement in Latex. Let's think step by step."""
        elif format == "test_nl_with_given_sstl_to_python" or format == "train_nl_with_given_sstl_to_python":
            return """Below is a natural language description of partial differential equation optimization problem, paired with your spatial-signal temporal logic description of the same problem provided earlier. Note that there may be mistakes in the spatial-signal temporal logic statement but the natural language description is accurate. Translate the problem into Python code following spatial-signal temporal logic."""
        elif format == "dpo_train_nl_to_sstl":
            return """Below is a natural language description of partial differential equation optimization problem. Instead of optimizing the provided problem directly, we want to optimize an intermediate problem to produce a state that will better serve to achieve the final conditions outlined in the natural language problem. Generate a spatial-signal temporal logic description in Latex code for such an intermediate problem."""
        elif format == "dpo_test_sstl_to_python":
            return """Below is a natural language description of partial differential equation optimization problem, paired with your spatial-signal temporal logic description of an intermediate problem provided earlier. Instead of optimizing the natural language problem directly, we want to optimize the intermediate problem to produce a state that will better serve to achieve the final conditions outlined in the natural language problem. Your spatial-signal temporal logic description in latex paired to the original problem describes this intermediate problem. Translate the intermediate problem into Python code following spatial-signal temporal logic."""
            # return """Below is a natural language description of partial differential equation optimization problem, paired with your spatial-signal temporal logic description of an intermediate problem provided earlier. Translate the intermediate problem into Python code following spatial-signal temporal logic."""
        
        # """Below is a natural language description of partial differential equation optimization problem, paired with your spatial-signal temporal logic description of an intermediate problem provided earlier. Instead of optimizing the provided problem directly, we want to optimize an intermediate problem to produce a state that will better serve to achieve the final conditions outlined in the natural language problem. Your spatial-signal temporal logic description paired to the original problem describes this intermediate problem. Translate the intermediate problem into Python code following spatial-signal temporal logic."""
        
            # try the prompt below if the above one doesn't work.

            # """Below is a natural language description of a partial differential equation optimization problem, paired with your spatial-signal temporal logic description of an intermediate problem provided earlier. Instead of optimizing the provided problem directly, we want to optimize this intermediate problem to produce a state that will better serve to achieve the final conditions outlined in the natural language problem. Use both the natural language description and the LaTeX code of the intermediate problem to translate the intermediate problem into Python code following spatial-signal temporal logic."""
        else:
            raise ValueError(f"Invalid format: {format}")