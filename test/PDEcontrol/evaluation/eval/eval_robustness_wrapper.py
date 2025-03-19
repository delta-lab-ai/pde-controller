# eval_robustness_wrapper.py
import sys
import json
try:
    from control.femformal.eval_robustness import eval_robustness as femformal_eval_robustness
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath('/localhome/mms43/scratch/mathcoder2/MathCoder2'))
    from control.femformal.eval_robustness import eval_robustness as femformal_eval_robustness


def main():
    nl_in_prompt = sys.argv[1]
    llm_output = sys.argv[2]
    robustness, runtime = femformal_eval_robustness(nl_in_prompt, llm_output)
    # Note that this code should not print anything else to stdout or it will screw with the evaluation
    print(json.dumps({
        "robustness": robustness,
        "runtime": runtime
        }))

if __name__ == "__main__":
    main()