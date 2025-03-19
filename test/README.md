# Testing

## Installation

Install conda environments `trainenv` and `pdecontrol` as specified in the [main README](../README.md0)

## Evaluation


We provide a test script for both zero-shot and few-shot evaluation on heat and wave PDEs used in our paper. Modify the settings in [`scripts/test_pdecontrol.sh`](./scripts/test_pdecontrol.sh).

Then to evaluate the models, set the paths in the following script and run it:

```bash
bash run_testing.sh`
```

Set `$MODELS` with the path to the directory where the model weights are stored, and set `$OUTPUT_DIR` with the path to the directory where you wish the inference results would be saved. This script would also create a `result` directory under `EVAL_DIR` where markdown files containing a table of all the results would be saved.


## Labels
The following code can be used to simulate labels for the ground-truth natural language & python inputs. To do so, run:

```bash
bash simulate_gt.sh
```