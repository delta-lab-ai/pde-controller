from .few_shot_prompting import FewShotPrompting

class FewShotTest(FewShotPrompting):
    def __init__(self, num_shots) -> None:
        super().__init__(num_shots=num_shots)

    def format_prompt(self, format, nl, sstl="", python=""):
        intruction = self.get_intruction(format)
        nl = nl.strip()
        sstl = sstl.strip()
        python = python.strip()
        if format == "nl_to_python":
            prompt = self.get_alpaca_format(intruction, task_input=nl, task_output=python, wrap_in_code_block='python')
            return prompt
        elif format == "test_nl_to_python_with_sstl_cot":
            # Don't wrap sstl in code block because the output is custom formatted
            if sstl != "" and python != "":
                # few shot prompt
                task_output = f"""Spatial Signal Temporal Logic:\n```latex\n{sstl}\n```\n\nPython:\n```python\n{python}\n```"""
            else:
                # actual test case
                task_output = ""
            prompt = self.get_alpaca_format(intruction, task_input=nl, task_output=task_output)
            return prompt
        elif format == "nl_to_sstl":
            
            task_output = f"""Spatial Signal Temporal Logic:\n```latex\n{sstl}"""
            if sstl != "":
                # few shot prompt
                task_output += """\n```"""
            prompt = self.get_alpaca_format(intruction, task_input=nl, task_output=task_output, wrap_in_code_block=None)
            return prompt
        elif format == "test_nl_with_given_sstl_to_python":
            task_input = f"""{nl}\n\nSpatial Signal Temporal Logic:\n```latex\n{sstl}\n```"""
            prompt = self.get_alpaca_format(intruction, task_input=task_input, task_output=python, wrap_in_code_block='python')
            return prompt


    def stop_words(self):
        return ["\n### Instruction:", "### Instruction:"]


