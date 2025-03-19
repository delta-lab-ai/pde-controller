import itertools
import os
import re
import shlex
import numpy as np
import regex
from copy import deepcopy
# try:
#     from eval.eval_utils import math_equal
#     from eval.ocwcourses_eval_utils import normalize_numeric, numeric_equality, normalize_symbolic_equation, SymbolicMathMixin
# except:
#     from eval_utils import math_equal
#     from ocwcourses_eval_utils import normalize_numeric, numeric_equality, normalize_symbolic_equation, SymbolicMathMixin

import math
import editdistance

import subprocess
import json

import operator
import functools
from bitarray import bitarray

def is_correct(pred, ans, prec=1e-3):
    if isinstance(pred, (int, float)) and isinstance(ans, (int, float)):
        return math.isclose(pred, ans, rel_tol=prec)
    if ans == 'timeout':
        return True
    return pred == ans



def eval_perplexity(item, logit_key='logits'):
    logits = item[logit_key]
    if logits is None:
        return "failed"

    sum_probs = sum(logits)
    entropy = - (1 / len(logits)) * sum_probs 
    return math.exp(entropy)

def eval_robustness_helper(robustness, natural_language, verbose = False):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    wrapper_script_path = os.path.join(current_dir, 'eval_robustness_wrapper.py')

    # conda_env, flag = 'path_to/conda_env/pdecontrol', '-p' #by path
    conda_env, flag = 'pdecontrol', '-n'   #by name
    command = f'conda run {flag} {conda_env} python {shlex.quote(wrapper_script_path)} {shlex.quote(natural_language)} {shlex.quote(robustness)}'

    process = subprocess.Popen(
        command,
        shell=True,
        executable='/bin/bash',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate()
    if verbose: print(stdout)
    if verbose: print(stderr)

    if process.returncode != 0:
        print(stderr)
        raise RuntimeError(f"Command failed with return code {process.returncode}")

    try:
        output = json.loads(stdout)
        pred_robustness = output.get("robustness", "failed")
        pred_runtime = output.get("runtime", "failed")
    except ValueError as e:
        raise ValueError(f"Failed to parse JSON output: {e}")
    
    if isinstance(pred_robustness, list):
        if len(pred_robustness) == 1 and isinstance(pred_robustness[0], int):
            return pred_robustness[0]
        else:
            raise ValueError("List must contain exactly one integer")
    return pred_robustness, pred_runtime

def eval_robustness(item, pred_key='predicted_python', nl_key='nl'):
    pred = item[pred_key]
    nl = item[nl_key]
    pred = pred.replace('\x00', '')
    nl = nl.replace('\x00', '')

    pred_robustness, pred_runtime = eval_robustness_helper(pred, nl)
    return pred_robustness, pred_runtime

def eval_robustness_gt(item, nl_key='nl', python_key = 'python', robustness_key='robustness', time_key='time'):
    gt_python = item[python_key]
    nl = item[nl_key]
    gt_robustness, gt_runtime = eval_robustness_helper(gt_python, nl)
    return gt_robustness, gt_runtime
    
def eval_robustness_DPO(item, pred_key='predicted_intermediate_python', nl_key='nl'):
    pred = item[pred_key]
    nl = item[nl_key]
    pred = pred.replace('\x00', '')
    nl = nl.replace('\x00', '')
    pred_robustness, pred_runtime = eval_robustness_helper(pred, nl)
    return pred_robustness, pred_runtime


def eval_edit_distance(item, python_key='python', pred_key='predicted_python'):
    return editdistance.eval(item[python_key], item[pred_key])

def eval_edit_distance_sstl(item, sstl_key='sstl', pred_key='predicted_sstl'):
    return eval_edit_distance(item=item, python_key=sstl_key, pred_key=pred_key)


# Define operators for comparison
ops = {
    '<=': operator.le,
    '<': operator.lt,
    '>=': operator.ge,
    '>': operator.gt,
    '=': operator.eq,
    '!=': operator.ne
}



# if the temporal operator is F, then the condition should evaluate as true for all points before and including the point that meets the condition. If it stops meeting the condition after then it should be false if it will never meet the condition again, and otherwise it should be true until and including the next point that meets the condition. Repeat for all time.
## For this reason we iterate through time BACKWARDS because as soon as a condition is met, all points before will be considered met also.

## For G we can also iterate backwards or forwards, it is the same

## For U we should iterate forwards because we need to check if the condition is met at the current point and all points before it.
## NOTE: we don't implement U yet

class OP_array():
    def __init__(self, operator) -> None:
        self.operator = operator
        self.array = bitarray()

    def insert(self, value : bool):
        raise NotImplementedError("This method should be implemented in the child class")

    def reset_state(self):
        raise NotImplementedError("This method should be implemented in the child class")
    
    def __and__(self, other):
        if isinstance(other, OP_array):
            return self.array & other.array
        elif isinstance(other, bitarray):
            return self.array & other
        else:
            raise TypeError("Unsupported operand type(s) for &: 'OP_array' and '{}'".format(type(other).__name__))

    def __rand__(self, other):
        if isinstance(other, bitarray):
            return other & self.array
        else:
            raise TypeError("Unsupported operand type(s) for &: '{}' and 'OP_array'".format(type(other).__name__))


    def __or__(self, other):
        if isinstance(other, OP_array):
            return self.array | other.array
        elif isinstance(other, bitarray):
            return self.array | other
        else:
            raise TypeError("Unsupported operand type(s) for |: 'OP_array' and '{}'".format(type(other).__name__))

    def __ror__(self, other):
        if isinstance(other, bitarray):
            return other | self.array
        else:
            raise TypeError("Unsupported operand type(s) for |: '{}' and 'OP_array'".format(type(other).__name__))
        
    def count(self):
        return self.array.count()

class G_array(OP_array):
    def __init__(self) -> None:
        super().__init__('G')
        self.index = 0
        self.flag = False
    
    def insert(self, value : bool, force=False):
        if force:
            self.array.append(value)
            return
        if value:
            if self.flag:
                # once we have appended false in this period, we should not append true again. "forall" failed.
                self.array.append(False)
            else:
                self.array.append(value)
        else:
            # if the value is false, then for every time step before this, the value should be false
            self.array[self.index:] = bitarray('0' * (len(self.array) - self.index))
            self.array.append(value)
            # for every time step in the future, append false:
            self.flag = True

    def reset_state(self):
        self.index = len(self.array)
        self.flag = False


class F_array(OP_array):
    def __init__(self) -> None:
        super().__init__('F')
        self.flag = False
    
    def insert(self, value : bool, force=False):
        if force:
            self.array.insert(0, value)
            return
        if value:
            # once we append true, all time steps before this should be true. Note that we are inserting backwards in time so no need to change the values before this
            self.flag = True
        if self.flag:
            self.array.insert(0, True)
        else:
            self.array.insert(0, value)

    def reset_state(self):
        self.flag = False

def op_array(operator):
    if operator == 'G':
        return G_array()
    elif operator == 'F':
        return F_array()
    else:
        raise NotImplementedError(f"Operator {operator} not implemented.")


def eval_IOU_DPO(item, sstl_key='w_sstl', pred_key='predicted_intermediate_sstl'):
    return eval_IOU(item, sstl_key=sstl_key, pred_key=pred_key)


def eval_IOU(item, sstl_key='sstl', pred_key='predicted_sstl'):
    #### Hyperparameters
    m_min, m_max, d_steps = -7, 7, 10
    b_min, b_max, d_steps = -500, 500, 50
    dt_steps, dx_steps = 50, 50
    ####
    dm = (m_max - m_min) / d_steps
    db = (b_max - b_min) / d_steps
    profile_parameters = {
        'm': (m_min, m_max, dm),
        'b': (b_min, b_max, db)
    }
    
    try:
        # split into clauses
        list_clauses_pred, junction_list_pred, bracket_tree_pred = parse_sstl(item[pred_key])
        list_clauses_gt, junction_list_gt, bracket_tree_gt = parse_sstl(item[sstl_key])
        # extract the clause components into list of dictionaries
        clauses_pred = [extract_sstl_clause_to_python(clause, i) for i, clause in list_clauses_pred]
        clauses_gt = [extract_sstl_clause_to_python(clause, i) for i, clause in list_clauses_gt]
    except:
        return "failed"
    
    # get the time and position global domains
    t_min, t_max = get_time_domain(clauses_gt, clauses_pred)
    x_min, x_max = get_pos_domain(clauses_gt, clauses_pred)

    dt = (t_max - t_min) / dt_steps
    dx = (x_max - x_min) / dx_steps

    # to assert that the bool arrays will be the same when iterating forwards and backwards:
    check_steps(t_min, t_max, dt)
    check_steps(x_min, x_max, dx)

    # iterate through time and space, fill in boolean arrays
    clauses_pred = iterate_through_time_space(clauses_pred, t_min, t_max, x_min, x_max, dt, dx, profile_parameters)
    clauses_gt = iterate_through_time_space(clauses_gt, t_min, t_max, x_min, x_max, dt, dx, profile_parameters)

    # element wise junction of the causes
    clause_junction_pred = compute_clause_junction(clauses_pred, junction_list_pred, bracket_tree_pred)
    clause_junction_gt = compute_clause_junction(clauses_gt, junction_list_gt, bracket_tree_gt)
        
    # compute the comparisons between the sstls
    iou = compute_iou(clause_junction_pred, clause_junction_gt)
    return iou


def parse_sstl(full_sstl):
    sstl_body = full_sstl

    conjunction_pattern = r'\\land'
    disjunction_pattern = r'\\lor'
    split_pattern = f'({conjunction_pattern}|{disjunction_pattern})'
    parts = re.split(split_pattern, sstl_body)

    clauses = []
    junctions = []
    brackets_tree = []
    stack = [brackets_tree]

    balance = 0
    for i, part in enumerate(parts):
        part = part.strip()
        if part in ['\\land', '\\lor']:
            if part == '\\land':
                junctions.append('and')
            elif part == '\\lor':
                junctions.append('or')
        else:
            clauses.append(part)
            open_parens = part.count('(')
            close_parens = part.count(')')
            if part[0] in ['G', 'F', 'U']:
                stack[-1].append(len(clauses) - 1)
            elif part[0] == '(':
                new_tuple = [len(clauses) - 1]
                stack[-1].append(new_tuple)
                stack.append(new_tuple)
            new_balance = balance + open_parens - close_parens
            if new_balance < balance:   # brackets were closed
                for _ in range(balance - new_balance):
                    if len(stack) > 1:
                        stack.pop()
                balance = new_balance
                

    def convert_to_tuple(structure):
        if isinstance(structure, list):
            return tuple(convert_to_tuple(item) for item in structure)
        return structure

    brackets_tree = convert_to_tuple(brackets_tree)
    return [(i, clause) for i, clause in enumerate(clauses)], junctions, brackets_tree


def parse_clause(statement):
    temporal_op_match = re.search(r'([FG])_\[\[([\d.]+),\s*([\d.]+)\]\]', statement)
    if not temporal_op_match:
        raise ValueError("Invalid temporal operator or time interval format")
    temporal_op = temporal_op_match.group(1)
    time_interval = [float(temporal_op_match.group(2)), float(temporal_op_match.group(3))]

    # position interval
    pos_interval_match = re.search(r'\\forall x \\in \[([\d.]+),\s*([\d.]+)\]', statement)
    if not pos_interval_match:
        raise ValueError("Invalid position interval format")
    pos_interval = [int(float(pos_interval_match.group(1))), int(float(pos_interval_match.group(2)))]
    # wxtract the expression, comparison operator, and threshold
    pattern1 = r'u\(x\)\s*(-\s*\(([-+]?\d*\.?\d+(?:e[-+]?\d+)?\s*\\cdot\s*x\s*\+\s*[-+]?\d*\.?\d+(?:e[-+]?\d+)?)\))?\s*(<=|>=|<|>|=|!=|=|\\leq|\\geq)\s*(-?\d+(\.\d+)?)'

    pattern2 = r'u\(x\)\s*(-\s*\(\s*([-+]?\s*x\s*/\(?\d+(?:e[-+]?\d+)?\)?\s*[-+]?\s*\d*\.?\d*(?:e[-+]?\d+)?\s*)\))?\s*(<=|>=|<|>|=|!=|\\leq|\\geq)\s*(-?\d+(\.\d+)?)'

    pattern3 = r'u\(x\)\s*(-\s*\(\s*([-+]?\s*x\s*\\cdot\s*[-+]?\s*\d*\.?\d+(?:e[-+]?\d+)?\s*[-+]?\s*[-+]?\d*\.?\d+(?:e[-+]?\d+)?\s*)\))?\s*(<=|>=|<|>|!=|\\leq|\\geq)\s*(-?\d+(\.\d+)?)'

    pattern4 = r'u\(x\)\s*(-\s*\(\s*([-+]?\s*x\s*\\cdot\s*\(?[-+]?\s*\d*\.?\d+(?:e[-+]?\d+)?\)?\s*[-+]?\s*\d*\.?\d*(?:e[-+]?\d+)?\s*)\)\s*)?\s*(<=|>=|<|>|!=|\\leq|\\geq)\s*(-?\d+(\.\d+)?)'

    pattern5 = r'u\(x\)\s*-\s*\(\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?x\s*[-+]\s*[-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*\)\s*(<=|>=|<|>|=|!=|\\leq|\\geq)\s*(-?\d+(\.\d+)?)'

    patterns = [pattern1, pattern2, pattern3, pattern4, pattern5]
    for i, p in enumerate(patterns):
        expression_match = re.search(p, statement)
        if expression_match:
            if i < 4:
                expression = expression_match.group(2) if expression_match.group(2) else "0"
                comparison_op = expression_match.group(3)
                threshold = expression_match.group(4)
            else:
                expression = expression_match.group(1) if expression_match.group(1) else "0"
                comparison_op = expression_match.group(2)
                threshold = expression_match.group(3)
            break
    
    if not expression_match:
        raise ValueError("Invalid expression format")

    comparison_op = comparison_op.replace('\\leq', '<=').replace('\\geq', '>=')
    # print(statement)
    # print("Temporal Operator:", temporal_op)
    # print("Time Interval:", time_interval)
    # print("Position Interval:", pos_interval)
    # print("Expression:", expression, clean_expression(expression))
    # print("Comparison Operator:", comparison_op)
    # print("Threshold:", threshold, float(clean_expression(remove_latex_syntax(threshold))))

    expression = clean_expression(expression)
    threshold = float(clean_expression(remove_latex_syntax(threshold)))
    return temporal_op, time_interval, pos_interval, expression, comparison_op, threshold


def remove_latex_syntax(expression):
    expression =expression.replace('\\left', '')
    expression = expression.replace('\\right', '')
    return expression

def clean_expression(expression):
    # Replace LaTeX-specific syntax with Python syntax
    expression = expression.replace('\\cdot', '*')
    expression =expression.replace('\\left', '(')
    expression = expression.replace('\\right', ')')
    # replace \\frac{}{} with /
    expression = re.sub(r'\\frac\{(.*?)\}\{(.*?)\}', r'\1/\2', expression)
    # Insert * between a number and a variable x
    expression = re.sub(r'(\d)([x])', r'\1*\2', expression)
    return expression


def check_steps(range_min, range_max, step, auto_adjust=True):
    if (range_max - range_min) % step != 0:
        if auto_adjust:
            # adjust the range to be divisible by the step
            range_max = range_max + (step - ((range_max - range_min) % step))
        else:
            raise ValueError(f"Step size {step} does not divide the range {range_min} to {range_max}")


def extract_sstl_clause_to_python(sstl_clause, index=None):
    temporal_op, time_interval, pos_interval, expression, comparison_op, threshold = parse_clause(sstl_clause)
    dict_clause = {
        'temporal_op': temporal_op,
        'time_interval': time_interval,
        'pos_interval': pos_interval,
        'expression': expression,
        'comparison_op': comparison_op,
        'threshold': threshold
    }
    if index is not None:
        dict_clause['index'] = index
    return dict_clause


def get_time_domain(clauses_dict_gt, clauses_dict_pred):
    t_min = min([clause['time_interval'][0] for clause in clauses_dict_gt + clauses_dict_pred])
    t_max = max([clause['time_interval'][1] for clause in clauses_dict_gt + clauses_dict_pred])
    return t_min, t_max

def get_pos_domain(clauses_dict_gt, clauses_dict_pred):
    x_min = min([clause['pos_interval'][0] for clause in clauses_dict_gt + clauses_dict_pred])
    x_max = max([clause['pos_interval'][1] for clause in clauses_dict_gt + clauses_dict_pred])
    return x_min, x_max


def check_conditions(bool_array, time_interval, pos_interval, clause, dt, dx, profile_parameters, direction='forwards'):
    """
    Iterates through the conditions in the specified direction (forwards or backwards) in time.
    Returns True if the conditions hold for the given time, spatial location, and u(x) value.
    """
    clause_time_interval = clause['time_interval']
    clause_pos_interval = clause['pos_interval']
    expression = clause['expression']
    comparison_op = clause['comparison_op']
    threshold = clause['threshold']

    params = get_profiles(profile_parameters, 'linear')
    if direction == 'forwards':
        time_range = np.arange(time_interval[0], time_interval[1] + dt + 1e-7, dt)
        pos_range = np.arange(pos_interval[0], pos_interval[1] + dx + 1e-7, dx)
    elif direction == 'backwards':
        time_range = np.arange(time_interval[1], time_interval[0] - dt - 1e-7, -dt)
        pos_range = np.arange(pos_interval[1], pos_interval[0] - dx - 1e-7, -dx)
    else:
        raise ValueError("Invalid direction. Use 'forwards' or 'backwards'.")

    for m, b in params:
        for time in time_range:
            u = m * time + b
            forall_pos_flag = None
            for pos in pos_range:
                if time >= clause_time_interval[0] and time <= clause_time_interval[1] and pos >= clause_pos_interval[0] and pos <= clause_pos_interval[1]:
                    # inside the domain, check expression
                    expr_value = evaluate_expression(expression, pos)
                    condition = ops[comparison_op](u - expr_value, threshold)
                    if condition:
                        # only set the forall flag to True if it has not been false yet.
                        if forall_pos_flag is None:
                            forall_pos_flag = True
                    else:
                        forall_pos_flag = False

            if forall_pos_flag is None:
                # outside the domain -> vacuously true
                bool_array.insert(True, force=True)
                s = 'v'
            elif forall_pos_flag == True:
                # inside the domain and the condition holds for all positions
                bool_array.insert(True)
                s = 't'
            else:
                # at some point inside the domain, the condition does not hold
                bool_array.insert(False)
                s = 'f'
        bool_array.reset_state()
    return bool_array

def iterate_through_time_space(clauses, t_min, t_max, x_min, x_max, dt, dx, profile_parameters):
    for i, clause in enumerate(clauses):
        temporal_op = clause['temporal_op']
        if temporal_op == 'F':
            bool_array = op_array('F')
            bool_array = check_conditions(bool_array, time_interval=[t_min, t_max], pos_interval=[x_min, x_max], clause=clause, dt=dt, dx=dx, profile_parameters=profile_parameters, direction='backwards')
        elif temporal_op == 'G' or temporal_op == 'U':
            bool_array = op_array(temporal_op)
            bool_array = check_conditions(bool_array, time_interval=[t_min, t_max], pos_interval=[x_min, x_max], clause=clause, dt=dt, dx=dx, profile_parameters=profile_parameters, direction='forwards')
        else:
            raise NotImplementedError(f"Temporal operator {temporal_op} not implemented.")
        clause['bool_array'] = bool_array
    return clauses

def get_profiles(profile_parameters, profile_type='linear'):
    if profile_type == 'linear':
        m_min, m_max, dm = profile_parameters['m']
        b_min, b_max, db = profile_parameters['b']
        m = np.arange(m_min, m_max, dm)
        b = np.arange(b_min, b_max, db)
        return itertools.product(m, b)
    else:
        raise NotImplementedError(f"Profile type {profile_type} not implemented.")

def evaluate_expression(expression, x):
    def u(x):
        return x
    expression = convert_expression_to_python(expression)
    return eval(expression)

def convert_expression_to_python(expression):
    expression = expression.replace('\\cdot', '*').replace('\\left', '(').replace('\\right', ')')
    return expression

def compute_clause_junction(clauses_list, junction_list, brackets_tree : tuple | int):
    """
    Computes the junction of the clauses based on the junction list and bracket structure.
    Assumes each clause has a bool_array attribute.
    """
    def evaluate_brackets(brackets):
        if isinstance(brackets, int):
            # base case: single clause index
            return clauses_list[brackets]['bool_array'].array
        elif isinstance(brackets, tuple):
            # recursive case: nested structure
            results = [evaluate_brackets(b) for b in brackets]            
            # determine the junction(s) to use
            junction_indices = find_junction_indices(brackets)
            result = results[0]
            for i in range(1, len(results)):
                junction = junction_list[junction_indices[i-1]]
                if junction == 'and':
                    result = results[i] & result
                elif junction == 'or':
                    result = results[i] | result
                else:
                    raise NotImplementedError(f"Junction {junction} not implemented.")
            return result
    
    def find_junction_indices(brackets):
        """find the correct junction indices for a given bracket"""
        flat_brackets = flatten_brackets(brackets_tree)
        indices = []
        
        def find_index(b):
            if isinstance(b, int):
                return flat_brackets.index(b)
            elif isinstance(b, tuple):
                # for nested tuples
                return flat_brackets.index(flatten_brackets(b)[-1])
        
        for b in brackets[:-1]:
            indices.append(find_index(b))
        return indices
    
    def flatten_brackets(brackets):
        """flatten the nested brackets structure"""
        if isinstance(brackets, int):
            return [brackets]
        elif isinstance(brackets, tuple):
            flat_list = []
            for b in brackets:
                flat_list.extend(flatten_brackets(b))
            return flat_list
    
    return evaluate_brackets(brackets_tree)


def compute_iou(clause_junction_pred, clause_junction_gt):
    # true positives
    intersection = clause_junction_pred & clause_junction_gt
    union = clause_junction_pred | clause_junction_gt
    if union.count() == 0:
        iou = 1
    else:
        iou = intersection.count() / union.count()
    return iou


def mean_and_std(data : list) -> tuple:
    # if len(data) == 0:
    #     raise ValueError("Empty data list")
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    return mean, std