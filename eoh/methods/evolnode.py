from abc import ABC, abstractmethod
import struct

import urllib
from .meta_prompt import CHOOSE_USEFUL_LINKS, GENERATE_NODES_FROM_API, MetaPrompt, PromptMode, parse_evol_response, spawn_test_cases
from .meta_prompt import MetaPlan, extract_json_from_text, extract_python_code, ALIGNMENT_CHECK_PROMPT, check_n_rectify_plan_dict
from .meta_execute import call_func_code, call_func_prompt_parallel, call_func_prompt, call_func_prompts, compile_code_with_references, combine_scores, combine_errors
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .llm import get_multiple_response, get_openai_response
import re, os, json, time
from tqdm import tqdm 
from collections import defaultdict
from typing import Optional, Union, Dict, List, Callable, Tuple
from collections import Counter
import requests
from bs4 import BeautifulSoup
import re
from typing import Any, Dict
import http 


MAX_ATTEMPTS = 6
SPAWN_TEST_MAX_TRIES = 20
NODE_EVOLVE_MAX_ATTEMPTS = 5
NODE_EVOLVE_BATCH_SIZE = 20

def map_input_output(test_case_list: List[dict], input_names: List[str], output_names: List[str]) -> List[Tuple[dict, dict]]:
    inputs = []
    outputs = []
    for test_case in test_case_list:
        input = {key: test_case['input'][key] for key in input_names}
        output = {key: test_case['expected_output'][key] for key in output_names}
        inputs.append(input)
        outputs.append(output)
    return list(zip(inputs, outputs))
    

def clean_str(s: str) -> str:
    def clean_line(line: str) -> str:
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, line)
        if not urls:
            return line.split("//")[0].split("#")[0] # Remove comments 
        else:
            return line
    return ('\n').join(map(clean_line, s.split('\n')))




def type_to_metric(value) -> Callable:
    
    def sudo_metric(x, y):
        return True, ""
    
    def integer_metric(x, y: int):
        """
        Expect integer value to be the same with each other
        """
        try:
            x_int = int(x)
        except:
            err_msg = f"Value {x} can't be converted into integer"
            return False, err_msg
        return abs(x_int - y) < 1, ""
    
    def float_metric(x, y: float):
        try:
            x_float = float(x)
        except:
            err_msg = f"Value {x} can't be converted into float"
            return False, err_msg
        return abs(x_float - y) <= 0.001, ""
    
    if isinstance(value, int): 
        return integer_metric
    elif isinstance(value, float):
        return float_metric
    else:
        return sudo_metric


def _require_llm_metric(value) -> bool:
    if isinstance(value, int): 
        return False
    elif isinstance(value, float):
        return False
    else:
        return True
    
def require_llm_metric(name: str, value: str, custom_metric_map: Optional[Dict[str, Callable]] = None) -> bool: 
    if custom_metric_map is not None and name in custom_metric_map:
        return False
    else:
        return _require_llm_metric(value)
    


def _check_alignment_with_metric(pred_output: dict, target_output: dict, custom_metric_map: Optional[Dict[str, Callable]] = None):
    """ 
    Input prediction & target dictionary, design algorithm for specific output value types' matching
    """
    error_msg = ""
    for key, target_value in target_output.items():
        if key not in pred_output:
            error_msg += f"Key {key} not found in prediction output\n"
            return False, error_msg
        
        pred_value = pred_output[key]
        if custom_metric_map is not None and key in custom_metric_map:
            metric = custom_metric_map[key]
        else:
            metric = type_to_metric(target_value)
        
        is_aligned, error_msg_delta = metric(pred_value, target_value)
        if not is_aligned:
            error_msg += error_msg_delta + "\n"
            error_msg += f"Value mismatch for key {key}: {pred_value} != {target_value}\n"
            return False, error_msg

    return True, ""


def _check_alignment_with_llm_sequential(pred_output: dict, target_output: dict, custom_metric_map: Optional[Dict[str, Callable]] = None,
                                          get_response: Optional[Callable] = get_openai_response, max_tries: int = 3):
    """ 
    Alignment checking with expected outputs with LLM
    Deprecated
    """
    error_msg = ""
    prompt = ALIGNMENT_CHECK_PROMPT.format(pred_output=pred_output, target_output=target_output)

    trimmed_pred_output = {k: v for k, v in pred_output.items() if require_llm_metric(k, v, custom_metric_map)}
    trimmed_target_output = {k: v for k, v in target_output.items() if require_llm_metric(k, v, custom_metric_map)}
    
    if len(trimmed_pred_output) == 0 and len(trimmed_target_output) == 0:
        return True, ""
    
    # Update the prompt with trimmed outputs
    prompt = ALIGNMENT_CHECK_PROMPT.format(pred_output=trimmed_pred_output, target_output=trimmed_target_output)
    for i in range(max_tries):
        try:
            response = get_response(prompt)
            check_dict = extract_json_from_text(response)
            if check_dict['aligned']:
                return True, ""
            else:
                return False, "--- LLM-Evaluation concludes prediction is not aligned with target output"
        except Exception as e:
            error_msg = "--- Parsing LLM Evaluation Failed: " + str(e)
            
    return False, error_msg



def _check_alignment_with_metric_parallel(output_per_code_per_test: Dict[int, Dict[int, Dict]],
                                          errors_per_code_per_test: Dict[int, Dict[int, List[str]]],
                                          test_inputs: List[Dict], 
                                          target_outputs: List[Dict],
                                          custom_metric_map: Optional[Dict[str, Callable]] = None):
    
    scores_per_code_per_test = defaultdict(lambda: defaultdict(list))
    for code_index in output_per_code_per_test:
        for test_index in output_per_code_per_test[code_index]:
            pred_output, target_output = output_per_code_per_test[code_index][test_index], target_outputs[test_index]
            
            trimmed_pred = {}
            trimmed_target = {}
            
            if isinstance(pred_output, dict) and isinstance(target_output, dict):
                trimmed_pred = {k: v for k, v in pred_output.items() if not require_llm_metric(k, v, custom_metric_map)}
                trimmed_target = {k: v for k, v in target_output.items() if not require_llm_metric(k, v, custom_metric_map)}            
            
            if len(trimmed_pred) == 0 and len(trimmed_target) == 0:
                continue
            
            is_aligned, error_msg = _check_alignment_with_metric(pred_output, target_output, custom_metric_map)
            scores_per_code_per_test[code_index][test_index].append(float(is_aligned))
            
            if not is_aligned:
                error_msg = f"Input: {test_inputs[test_index]}, prediction is not aligned with expected output, Expected: {target_outputs[test_index]} Predicted: {pred_output}, Error message: {error_msg}\n"
                errors_per_code_per_test[code_index][test_index].append(error_msg)
            
    return scores_per_code_per_test, errors_per_code_per_test


def _check_alignment_with_llm_parallel(output_per_code_per_test: Dict[int, Dict[int, Dict]], 
                                       errors_per_code_per_test: Dict[int, Dict[int, List[str]]],
                                       test_inputs: List[Dict], 
                                       target_outputs: List[Dict], 
                                       get_response: Optional[Callable] = get_openai_response, 
                                       batch_size: int = 3,
                                       custom_metric_map: Optional[Dict[str, Callable]] = None):
    
    # Unpack output dictionary into prompts
    prompts = []
    indices = []
    for code_index in output_per_code_per_test:
        for test_index in output_per_code_per_test[code_index]:
            pred_output, target_output = output_per_code_per_test[code_index][test_index], target_outputs[test_index]
            
            trimmed_pred = {}
            trimmed_target = {}
            for target_name, target_value in target_output.items():
                if require_llm_metric(target_name, target_value, custom_metric_map) and target_name in pred_output:
                    trimmed_target[target_name] = target_value
                    trimmed_pred[target_name] = pred_output[target_name]
    
            if len(trimmed_pred) == 0 and len(trimmed_target) == 0:
                continue
            
            prompt = ALIGNMENT_CHECK_PROMPT.format(pred_output=trimmed_pred, target_output=trimmed_target)
            prompts.append(prompt)
            indices.append((code_index, test_index))
            
            
    prompts = prompts * batch_size
    indices = indices * batch_size
            
    # Get LLM responses
    desc_str = f"Running LLM-based alignment check in parallel with batch size {len(prompts)}"
    responses = get_response(prompts, desc=desc_str)

    # Pack response into scores dictionary
    scores_per_code_per_test = defaultdict(lambda: defaultdict(list))
    for i, response in enumerate(responses):
        code_index, test_index = indices[i]
        try:
            check_dict = extract_json_from_text(response)
            scores_per_code_per_test[code_index][test_index].append(float(check_dict['aligned']))
        except:
            errors_per_code_per_test[code_index][test_index].append(f"LLM Evaluation Failed: {response}")
    
    # Calculate average score and report issue summary
    score_per_code_per_test = defaultdict(lambda: defaultdict(float))

    for code_index in scores_per_code_per_test:
        for test_index in scores_per_code_per_test[code_index]:
            score = sum(scores_per_code_per_test[code_index][test_index]) / len(scores_per_code_per_test[code_index][test_index])
            score_per_code_per_test[code_index][test_index] = score
            if score < 0.8: # print err message | threshold of adding error message
                error_msg = f"Input: {test_inputs[test_index]}, prediction is not aligned with expected output, Expected: {target_outputs[test_index]} Predicted: {output_per_code_per_test[code_index][test_index]}\n"
                errors_per_code_per_test[code_index][test_index].append(error_msg)
            
    return score_per_code_per_test, errors_per_code_per_test



def check_alignment_sequential(pred_output: dict, target_output: dict, get_response: Optional[Callable] = get_openai_response, max_tries: int = 3):
    """ 
    Alignment checking with expected outputs with LLM
    - we have two dictionary, need to make sure they are aligned
    - 1. they have same keys 
    - 2. their values are 'basically the same'
    
    Fix: For integer output, use exact match metric instead for alignment check
    """
    error_msg = ""
    
    # 1. Metric-based alignment check 
    _, error_msg_delta = _check_alignment_with_metric(pred_output, target_output)
    error_msg += error_msg_delta
    
    if error_msg != "":
        return False, error_msg
    
    # 2. LLM-based alignment check 
    _, error_msg_delta = _check_alignment_with_llm_sequential(pred_output, target_output, get_response, max_tries)
    error_msg += error_msg_delta
    
    
    if error_msg == "":
        return True, ""
    else:
        return False, error_msg
    
    
def check_alignment_parallel(output_per_code_per_test: Dict[int, Dict[int, Dict]], test_inputs: List[Dict], target_outputs: List[Dict], get_response: Optional[Callable] = get_openai_response, 
                             batch_size: int = 3, custom_metric_map: Optional[Dict[str, Callable]] = None):
    
    # make sure to combined score obtained from both steps (average of both)
    errors_per_code_per_test = defaultdict(lambda: defaultdict(list))
    
    # Get scores from metric-based alignment check
    metric_scores, errors_per_code_per_test = _check_alignment_with_metric_parallel(output_per_code_per_test, errors_per_code_per_test, test_inputs, target_outputs, custom_metric_map)
    
    # Get scores from LLM-based alignment check 
    llm_scores, errors_per_code_per_test = _check_alignment_with_llm_parallel(output_per_code_per_test, errors_per_code_per_test, test_inputs, target_outputs,
                                                                              get_response, batch_size)
    
    score_per_code_per_test = combine_scores(llm_scores, metric_scores)
    
    return score_per_code_per_test, errors_per_code_per_test
    
    
    
    
def print_individual_info(code_index: int, fitness: float, err_msg: str, reasoning: str, code: str):
    print("\n" + "="*80)  # Top separator
    print(f"📊 Code {code_index}: Fitness: {fitness*100:.1f}%")
    if err_msg.strip():  # Only print error section if there are errors
        print("-"*80)  # Separator between sections
        print(f"❌ Error Messages:\n{err_msg}")
    print("="*80 + "\n")  # Bottom separator
        
    
class Fitness: 
    
    def __init__(self, structural_fitness: float, functional_fitness: float):
        self.structural_fitness = structural_fitness
        self.functional_fitness = functional_fitness
        
    def __call__(self):
        return (self.structural_fitness + self.functional_fitness) / 2
    
    def __str__(self):
        if self.structural_fitness <= 0.2:
            return "This node has no idea of what's going on"
        if self.structural_fitness >= 0.8 and self.functional_fitness <= 0.2:
            return "This function runs, but never correctly"
        return f"Function runs with success rate: {self.structural_fitness*100:.1f}%, runs correctly with rate: {self.functional_fitness*100:.1f}%"
        
    

#####################################
#        Evolution on Graph         #
#-----------------------------------#
# - Node: Code+Compiler, Prompt+LLM #
# - Node tries to complete task     #
# - PlanNode breaks down tasks      #
#####################################

# Focus: 
# - Node retrieve relevant node from Node database 
# - Node tries to complete, if failed, a PlanNode helps breaking down the task 
# - Nodes are then created for each sub-task. 
# - Game on ....
# - No gradient info available, genetic search is adopted here. 

class EvolNode:
    
    def __init__(self, meta_prompt: MetaPrompt, 
                 code: Optional[str] = None, 
                 reasoning: Optional[str] = None,
                 get_response: Optional[Callable] = get_openai_response, 
                 test_cases: Optional[List[Tuple[Dict, Dict]]] = None,
                 custom_metric_map: Optional[Dict[str, Callable]] = None,
                 libary_dir: str = "methods/nodes/",
                 fitness: float = 0.0):
        """ 
        Executable Task
        """
        self.code = code
        self.reasoning = reasoning
        self.fitness = fitness
        self.meta_prompt = meta_prompt
        self.test_cases = []
        self._get_response = get_response
        self.library_dir = libary_dir
        self.relevant_nodes = []
        self.error_msg = "" # contains information about encountered error :: TBD :: use LLM to summarize it
        self.custom_metric_map = custom_metric_map
        
        if test_cases is not None:
            self.test_cases = test_cases
        else:
            self.get_test_cases(3) # generate 3 test cases for new node
            
        
    def get_response(self, prompt: str, desc: str = ""):
        if isinstance(prompt, str):
            prompts = [prompt]
            n_prompt = 1
        else:
            prompts = prompt
            n_prompt = len(prompts)
        try:
            responses = self._get_response(prompts, desc)
        except Exception as e:
            print(e.with_traceback(e.__traceback__))
            responses = []
            for p in prompts:
                responses.append(self._get_response(p))
        if n_prompt == 1 and isinstance(prompt, str):
            return responses[0]
        else:
            return responses
    
    def _get_extend_test_cases_response(self, num_cases: int = 1, feedback: str = ""):
        if feedback != "":
            eval_prompt = self.meta_prompt._get_eval_prompt_with_feedback(num_cases, feedback)
        else:
            eval_prompt = self.meta_prompt._get_eval_prompt(num_cases)
        response = self.get_response(eval_prompt)
        self.temp_response = response # added info for debugging
        response = clean_str(response) # extra cleaning to enhance robustness
        return response
        
    def _extend_test_cases(self, num_cases: int = 5, feedback: str = ""):
        response = self._get_extend_test_cases_response(num_cases, feedback)
        
        try:
            test_case_list = extract_json_from_text(response)
            self.test_cases.extend(map_input_output(test_case_list, self.meta_prompt.inputs, self.meta_prompt.outputs))
            self._filter_test_cases()
        except Exception as e:
            print("TEST CASE PARSING ERROR")
            print(e, e.__traceback__)# alas, response parsing failed
        
    def _filter_test_cases(self):
        seen_values = defaultdict(set)
        filtered_cases = []
        for case_tuple in self.test_cases:
            case_input = case_tuple[0]
            is_unique = True
            for key, value in case_input.items():
                if value in seen_values[key]:
                    is_unique = False
                    break
                seen_values[key].add(value)
            if is_unique:
                filtered_cases.append(case_tuple)
        self.test_cases = filtered_cases
        
    def get_test_cases(self, num_cases: int = 100, feedback: str = ""):
        """ 
        Generate test cases for current node
        """
        batch_case_amount = min(20, num_cases)
        max_attemp = 3 + int(num_cases / batch_case_amount)     
        curr_attemp = 1
        with tqdm(total=num_cases, desc="Generating test cases", unit="case") as pbar:
            curr_progress = 0
            while num_cases > len(self.test_cases) and curr_attemp <= max_attemp:
                generate_amount = min(batch_case_amount, num_cases - len(self.test_cases))
                self._extend_test_cases(generate_amount, feedback)
                new_progress = len(self.test_cases)
                pbar.update(new_progress - curr_progress)  # Update progress bar with new test cases
                curr_progress = new_progress
                curr_attemp += 1
        print(f"--- Generated {len(self.test_cases)} test cases")
        return self.test_cases
    
    @property 
    def test_inputs(self):
        return [case[0] for case in self.test_cases]
    
    
    def _get_evolve_response_sequential(self, method: str, parents: Optional[list] = None, feedback: str = ""):
        prompt_method = getattr(self.meta_prompt, f'_get_prompt_{method}')
        prompt_content = prompt_method(parents)
        prompt_content += self.relevant_node_desc
        prompt_content += "\nIdea: " + feedback # External Guidance (perhaps we should reddit / stackoverflow this thingy)
     
        response = self.get_response(prompt_content)
        return response 
    
    def _get_evolve_prompt(self, method: str, parents: Optional[list] = None, feedback: str = ""):
        prompt_method = getattr(self.meta_prompt, f'_get_prompt_{method}')
        prompt_content = prompt_method(parents)
        prompt_content += self.relevant_node_desc
        prompt_content += "\nFeedback: " + feedback # External Guidance (perhaps we should reddit / stackoverflow this thingy)
        return prompt_content
    
    def _get_evolve_response(self, method: str, parents: Optional[list] = None, feedback: str = "", batch_size: int = 5):
        prompt_content = self._get_evolve_prompt(method, parents, feedback)
        prompts = [prompt_content] * batch_size
        desc_str = f"Running evolution strategy {method} in parallel with batch size {batch_size}" # Added description string for progress bar
        responses = self.get_response(prompts, desc=desc_str)
        return responses

    def _evolve(self, method: str, parents: list = None, feedback: str = "", batch_size: int = 5):
        """
        Note: Evolution process will be decoupled with the fitness assignment process
        """
        responses = self._get_evolve_response(method, parents, feedback, batch_size)
        reasonings, codes = [], []
        for response in responses:
            try:
                reasoning, code = parse_evol_response(response)
                code = compile_code_with_references(code, self.referrable_function_dict) # deal with node references
                reasonings.append(reasoning)
                codes.append(code)
            except Exception as e:
                print('ERROR PARSING CODE')  
            
        return reasonings, codes
    
    def evolve(self, method: str, parents: list = None, replace=True, feedback: str = "", batch_size: int = 5, fitness_threshold: float = 0.8, 
               num_runs: int = 5, max_tries: int = 3, print_summary: bool = True, query_node: bool = True, timeout: bool = True):
        """
        Evolve node and only accept structurally fit solutions
        Attempts multiple evolutions before returning the final output
        """
        start_time = time.time()
        # Query once
        offsprings = [] 
        if query_node:
            self.query_nodes(ignore_self=replace, self_func_name=self.meta_prompt.func_name)
        query_end_time = time.time()
        query_time = query_end_time - start_time
        print(f"     :: Query time: {query_time:.2f}s")
        
        
        # Evolve many times
        reasonings, codes = self._evolve(method, parents, feedback=feedback, batch_size=batch_size)
        evolve_end_time = time.time()
        evolve_time = evolve_end_time - query_end_time
        print(f"     :: Evolution time: {evolve_time:.2f}s")
        
        self.reasonings = reasonings
        self.codes = codes
        fitness_per_code, errors_per_code, global_summary = self._evaluate_fitness(codes=codes, max_tries=max_tries, num_runs=num_runs, custom_metric_map=self.custom_metric_map, timeout=timeout)
        end_time = time.time()
        evaluation_time = end_time - evolve_end_time
        if print_summary:
            generation_time = evolve_time
            total_time = end_time - start_time
            print(global_summary + 
                  f"  ⏱️ Time breakdown:\n" + 
                  f"     :: Query time: {query_time:.2f}s\n" +
                  f"     :: Evolution time: {evolve_time:.2f}s\n" +
                  f"     :: Evaluation time: {evaluation_time:.2f}s\n" +
                  f"     :: Total time: {total_time:.2f}s\n")
                
        for code_index in fitness_per_code:
            fitness = fitness_per_code[code_index]()
            reasoning = reasonings[code_index]
            code = codes[code_index]
            err_msg = "\n".join(str(err) for err in errors_per_code[code_index]) if len(errors_per_code[code_index]) > 0 else ""
            if print_summary:
                print_individual_info(code_index, fitness, err_msg, reasoning, code)
            
            if fitness >= self.fitness:
                if replace:
                    self.reasoning, self.code = reasoning, code
                    self.fitness = fitness
                    self.error_msg = err_msg

            # if fitness > fitness_threshold:
            offsprings.append({"reasoning": reasoning, "code": code, "fitness": fitness, "err_msg": err_msg})
                
            if fitness >= 1.0:
                self.save(self.library_dir)
            
        if not replace: 
            return offsprings
        
    
    def _evaluate_structure_fitness(self, test_inputs: List[Dict], code: Optional[str] = None) -> Tuple[float, str]:
        """ 
        Check for compilation sucess, type consistency
        """
        total_tests = len(test_inputs)
        passed_tests = 0

        if code is None:
            code = self.code

        error_msg = ""
        for test_input in test_inputs:
        
            if self.meta_prompt.mode == PromptMode.CODE:   
                _, error_msg_delta = self.call_code_function(test_input, code, file_path=None)
                if error_msg_delta == "":
                    passed_tests += 1
                else:
                    error_msg += error_msg_delta
                    
            elif self.meta_prompt.mode == PromptMode.PROMPT:
                _, error_msg_delta = self.call_prompt_function(test_input, code, self.get_response)
                if error_msg_delta == "":
                    passed_tests += 1
                else:
                    error_msg += error_msg_delta
            else:
                raise ValueError(f"Unknown mode: {self.meta_prompt.mode}")

        return passed_tests / total_tests, error_msg
    
    def call_prompt_function(self, test_input: Dict, code: Optional[str] = None, max_tries: int = 3): # TBD: Batch inference and pick the one which works
        if code is None:
            code = self.code
        
        error_msg = set()
        for i in range(max_tries):
            output_dict, error_msg_delta = call_func_prompt(test_input, code, self.get_response)
            if error_msg_delta == "":
                return output_dict, ""
            else:
                error_msg.add(error_msg_delta)
        error_msg = "--- Calling Prompt Function Error:\n" + "\n".join(list(error_msg))
        return None, error_msg    
    
    
    def call_prompt_functions(self, test_input: Dict, code: Optional[str] = None, max_tries: int = 3):
        if code is None:
            code = self.code 
            
        error_msg = set()
        output_dict, error_msg_delta = call_func_prompts(test_input, code, self.get_response, max_tries)
        if error_msg_delta == "":
            return output_dict, ""
        else:
            error_msg.add(error_msg_delta)
        error_msg = "--- Calling Prompt Function Error:\n" + "\n".join(list(error_msg))
        return None, error_msg     
        
    
    def call_code_function(self, test_input: Dict, code: Optional[str] = None, file_path: Optional[str] = None):
        if code is None:
            code = self.code
        
        output_value, error_msg = call_func_code(test_input, code, self.meta_prompt.func_name, file_path=file_path)
        output_name = self.meta_prompt.outputs[0]
        output_dict = {output_name: output_value}
        return output_dict, error_msg
    
    
    def call_prompt_function_parallel(self, test_inputs: List[Dict], codes: Optional[List[str]] = None, max_tries: int = 3):
        if codes is None:
            codes = [self.code]
        return call_func_prompt_parallel(test_inputs, codes, max_tries, self.get_response)
    
    
    def call_code_function_parallel(self, test_inputs: List[Dict], codes: Optional[List[str]] = None, file_path: Optional[str] = None, timeout: bool = True):
        output_per_code_per_test = defaultdict(lambda: defaultdict(dict))
        errors_per_code_per_test = defaultdict(lambda: defaultdict(list))
        
        if codes is None: 
            codes = [self.code]
        
        for (test_index, test_input) in enumerate(test_inputs):
            for (code_index, code) in enumerate(codes):
                output_value, error_msg = call_func_code(test_input, code, self.meta_prompt.func_name, file_path=file_path, timeout=timeout)
                if error_msg != "":
                    errors_per_code_per_test[code_index][test_index].append(error_msg)
                else:
                    output_name = self.meta_prompt.outputs[0]
                    output_dict = {output_name: output_value}
                    output_per_code_per_test[code_index][test_index] = output_dict
                            
        return output_per_code_per_test, errors_per_code_per_test
    
    def summarize_fitness(self, codes, score_per_code_per_test, output_per_code_per_test, test_inputs, max_tries):
        
        structual_fitness_per_code_per_test = defaultdict(lambda: defaultdict(float))
        for (code_index, code) in enumerate(codes):
            for (test_index, test_input) in enumerate(test_inputs):
                if code_index in output_per_code_per_test and test_index in output_per_code_per_test[code_index]:
                    list_output = list(output_per_code_per_test[code_index][test_index].values())
                    if self.meta_prompt.mode == PromptMode.CODE: 
                        structural_fitness = float(len(list_output) > 0)
                    elif self.meta_prompt.mode == PromptMode.PROMPT:
                        structural_fitness = float(len(list_output) / max_tries)
                else:
                    structural_fitness = 0
                structual_fitness_per_code_per_test[code_index][test_index] = structural_fitness
        
        structual_fitness_per_code = defaultdict(float)
        for code_index in structual_fitness_per_code_per_test:
            structual_fitness_per_code[code_index] = sum(list(structual_fitness_per_code_per_test[code_index].values())) / len(test_inputs)
        
        functional_fitness_per_code_per_test = defaultdict(lambda: defaultdict(float))
        for (code_index, code) in enumerate(codes):
            for (test_index, test_input) in enumerate(test_inputs):
                if code_index in score_per_code_per_test and test_index in score_per_code_per_test[code_index]:
                    functional_fitness = score_per_code_per_test[code_index][test_index]
                else:
                    functional_fitness = 0
                functional_fitness_per_code_per_test[code_index][test_index] = functional_fitness
        
        functional_fitness_per_code = defaultdict(float)
        for code_index in functional_fitness_per_code_per_test:
            functional_fitness_per_code[code_index] = sum(list(functional_fitness_per_code_per_test[code_index].values())) / len(test_inputs)
        
        fitness_per_code = defaultdict(Fitness)
        for code_index in structual_fitness_per_code:
            fitness_per_code[code_index] = Fitness(structual_fitness_per_code[code_index], functional_fitness_per_code[code_index])
        return fitness_per_code
    
    
    def _evaluate_fitness(self, test_cases: Optional[List[Tuple[Dict, Dict]]] = None, codes: Optional[List[str]] = [], 
                           max_tries: int = 3, num_runs: int = 1, custom_metric_map: Optional[Dict[str, Callable]] = None, timeout: bool = True) -> Fitness:
        
        """ 
        TBD: Parallel evaluation of all test cases
        """
        
        if len(codes) == 0: 
            codes = [self.code] 
        
        if test_cases is None:
            test_cases = self.test_cases
        
        if self.meta_prompt.mode == PromptMode.PROMPT:
            num_runs = min(2, num_runs) # sanity check against stochastic nature of prompt-based node
                
        test_inputs = [case[0] for case in test_cases]
        
        if self.meta_prompt.mode == PromptMode.CODE: 
            output_per_code_per_test, errors_per_code_per_test = self.call_code_function_parallel(test_inputs, codes, timeout=timeout)
        elif self.meta_prompt.mode == PromptMode.PROMPT:
            output_per_code_per_test, errors_per_code_per_test = self.call_prompt_function_parallel(test_inputs, codes, max_tries)
        else:
            raise ValueError(f"Unknown mode: {self.meta_prompt.mode}")
        
        # Print info about outputs and errors per code
        for code_index in output_per_code_per_test:
            print(f"\nCode {code_index} outputs:")
            for test_index in output_per_code_per_test[code_index]:
                print(f"Test {test_index}: {output_per_code_per_test[code_index][test_index]}")
                if errors_per_code_per_test[code_index][test_index]:
                    print(f"Errors: {errors_per_code_per_test[code_index][test_index]}")
        
        # alignment checking
        test_inputs = [case[0] for case in test_cases]
        target_outputs = [case[1] for case in test_cases]
        score_per_code_per_test, evaluate_errors_per_code_per_test = check_alignment_parallel(output_per_code_per_test, test_inputs, target_outputs, 
                                                                                              self.get_response, batch_size=num_runs, custom_metric_map=custom_metric_map)
        
        errors_per_code = combine_errors(evaluate_errors_per_code_per_test, errors_per_code_per_test)
        
        fitness_per_code = self.summarize_fitness(codes, score_per_code_per_test, output_per_code_per_test, test_inputs, max_tries)
        
        # Get best fitness
        best_fitness = max(fitness_per_code.values(), key=lambda x: x(), default=Fitness(0.0, 0.0))
        global_summary = (
            f"🏆 Best Code Performance Summary 🏆\n"
            f"  ⚡ Structural fitness: {best_fitness.structural_fitness:.2f}\n"
            f"  🎯 Functional fitness: {best_fitness.functional_fitness:.2f}\n" 
            f"  ⭐ Global fitness:     {best_fitness():.2f}\n"
            f"  🔄 Compiled solutions:        {len(codes)}\n"
        )
                
        return fitness_per_code, errors_per_code, global_summary  # TBD: return error messages ...
        
    
    def _evaluate_fitness_sequential(self, test_cases: Optional[List[Tuple[Dict, Dict]]] = None, code: Optional[str] = None, 
                                        max_tries: int = 3, num_runs: int = 1) -> Fitness:
        """ 
        Alignment checking with expected outputs with LLM
        """
        if code is None:
            return Fitness(0.0, 0.0), ""
        
        if test_cases is None:
            test_cases = self.test_cases
        
        if self.meta_prompt.mode == PromptMode.PROMPT:
            num_runs = min(2, num_runs) # sanity check against stochastic nature of prompt-based node
            
        test_cases = test_cases * num_runs # repeat
        
        total_tests = len(test_cases)
        compiled_tests = 0
        passed_tests = 0

        if code is None:
            code = self.code

        error_msg = ""
        issue_summary = ""
        for test_input, test_output in tqdm(test_cases, desc="Evaluating fitness"):
            
            if self.meta_prompt.mode == PromptMode.CODE:
                
                output_dict, error_msg_delta = self.call_code_function(test_input, code, file_path=None)
                
                if error_msg_delta == "":
                    compiled_tests += 1
                    # Issue seems to be empty output_dict obtained in here. It might be that the code function deliberately returns None
                    is_aligned, error_msg_delta2 = check_alignment_sequential(output_dict, test_output, self.get_response)
                    
                    if error_msg_delta2 != "":
                        error_msg += error_msg_delta2
                    elif is_aligned:
                        passed_tests += 1
                    else:
                        issue_summary += f"Input: {test_input}, Wrong Prediction: {output_dict}, Expected: {test_output}\n"
                else:
                    error_msg += error_msg_delta
                    issue_summary += f"Input: {test_input}, Output is missing or of wrong type, Expected: {test_output}\n"


            elif self.meta_prompt.mode == PromptMode.PROMPT:
                
                output_dict, error_msg_delta = self.call_prompt_function(test_input, code, max_tries)
                if error_msg_delta == "":
                    compiled_tests += 1
                    is_aligned = check_alignment_sequential(output_dict, test_output, self.get_response)
                    if is_aligned:
                        passed_tests += 1
                    if not is_aligned:
                        issue_summary += f"Input: {test_input}, Wrong Prediction: {output_dict}, Expected: {test_output}\n"
                else:
                    error_msg += error_msg_delta
                    issue_summary += f"Input: {test_input}, Output is missing or of wrong type, Expected: {test_output}\n"

            else:
                raise ValueError(f"Unknown mode: {self.meta_prompt.mode}")
            
        # add overal information
        global_summary = f"--- Compiled {compiled_tests} out of {total_tests} test cases\n"
        global_summary += f"--- Passed {passed_tests} out of {total_tests} test cases\n"
        issue_summary = global_summary + issue_summary

        structural_fitness = compiled_tests / total_tests
        functional_fitness = passed_tests / total_tests
        fitness = Fitness(structural_fitness, functional_fitness)
        
        return fitness, f" {str(fitness)}\n" + issue_summary + "\nError Message:\n" + error_msg


    def i1(self):
        return self._evolve('i1')
    
    def e1(self, parents: list):
        return self._evolve('e1', parents)
    
    def e2(self, parents: list):
        return self._evolve('e2', parents)
    
    def m1(self, parents: list):
        return self._evolve('m1', parents)
    
    def m2(self, parents: list):
        return self._evolve('m2', parents)
    
    def _rectify_input_dict_names(self, input_dict: dict):
        input_types = self.meta_prompt.input_types
        if (len(input_types) == len(list(set(input_types)))):
            new_input = {}
            param_names = self.meta_prompt.inputs
            for param_name, value in input_dict.items():
                if param_name in param_names:
                    new_input[param_name] = value
                else:
                    for idx, type_hint in enumerate(input_types):
                        if type(value).__name__ == type_hint:
                            new_input[param_names[idx]] = value
            input_dict = new_input    
        return input_dict
    
    def _rectify_input_names(self, inputs: Union[dict, list]):
        if isinstance(inputs, dict):
            return self._rectify_input_dict_names(inputs)
        elif isinstance(inputs, list):
            return [self._rectify_input_dict_names(input_dict) for input_dict in inputs]
    
    def batch_inference(self, inputs: list, codes: Optional[List[str]] = None, max_tries: int = 3, timeout: bool = True):
        """ 
        Should be used in Batch Evaluation
        """

        # Input name rectification        
        inputs = self._rectify_input_names(inputs)
        
        if codes is None:
            codes = [self.code]

        # Batch inference 
        if self.meta_prompt.mode == PromptMode.CODE:
            output_per_code_per_test, errors_per_code_per_test = self.call_code_function_parallel(inputs, codes, timeout=timeout)
        elif self.meta_prompt.mode == PromptMode.PROMPT:
            output_per_code_per_test, errors_per_code_per_test = self.call_prompt_function_parallel(inputs, codes, max_tries)
        else:
            raise ValueError(f"Unknown mode: {self.meta_prompt.mode}")
        
        return output_per_code_per_test, errors_per_code_per_test
        
    
    def __call__(self, inputs, max_attempts: int = 3, batch_inference: bool = True, timeout: bool = True):
        """ 
        TBD: Inheritance to accumulated codebase with 'file_path' | Graph Topology naturally enables inheritance
        TBD: Stricter input / output type checking to ensure composibility
        """
            
        inputs = self._rectify_input_names(inputs)
            
        if self.meta_prompt.mode == PromptMode.CODE:
            output_value, err_msg = call_func_code(inputs, self.code, self.meta_prompt.func_name, file_path=None, timeout=timeout) # TODO: extend to multiple outputs ...
            output_name = self.meta_prompt.outputs[0]
            return {output_name: output_value} # assuming single output for code-based node
        
        elif self.meta_prompt.mode == PromptMode.PROMPT:
            output_name = self.meta_prompt.outputs[0]
            
            if batch_inference:
                output_dict, err_msg = self.call_prompt_functions(inputs, self.code, max_tries=max_attempts)
            else:
                output_dict, err_msg = self.call_prompt_function(inputs, self.code, max_tries=max_attempts)
            
            if output_dict is None or output_name not in output_dict:
                raise ValueError(err_msg)
            else:
                return {output_name: output_dict[output_name]}
    
    def save(self, library_dir: str = "methods/nodes/", overwrite=False) -> None:
        node_data = {
            "code": self.code,
            "reasoning": self.reasoning,
            "meta_prompt": self.meta_prompt.to_dict(),  # Assuming MetaPrompt has a to_dict method
            "test_cases": [{"input": test_case[0], "expected_output": test_case[1]} for test_case in self.test_cases],
            "fitness": self.fitness
        }
        node_path = os.path.join(library_dir, f"{self.meta_prompt.func_name}_node.json")
        os.makedirs(os.path.dirname(node_path), exist_ok=True)
        if overwrite or not os.path.exists(node_path):
            with open(node_path, 'w') as f:
                json.dump(node_data, f, indent=2)

    @classmethod 
    def load(cls, node_name: str, library_dir: str = "methods/nodes/", get_response: Optional[Callable] = get_openai_response) -> 'EvolNode':
        node_path = os.path.join(library_dir, f"{node_name}_node.json")
        with open(node_path, 'r') as f:
            node_data = json.load(f)
        meta_prompt = MetaPrompt.from_dict(node_data['meta_prompt'])  # Assuming MetaPrompt has a from_dict method
        test_cases = [tuple([test_case['input'], test_case['expected_output']]) for test_case in node_data['test_cases']]
        node = cls(meta_prompt=meta_prompt, code=node_data['code'], reasoning=node_data['reasoning'], test_cases=test_cases,
                   get_response=get_response, fitness=node_data['fitness'])
        return node
    
    def query_nodes(self, top_k: int = 5, ignore_self: bool = False, self_func_name: str = None) -> List['EvolNode']:
        """ 
        Query nodes from library
        """
        query_engine = QueryEngine(ignore_self=ignore_self, self_func_name=self_func_name)
        self.relevant_nodes = query_engine.query_node(self.meta_prompt.task)
        
    @property 
    def relevant_node_desc(self):
        if len(self.relevant_nodes) == 0:
            return ""
        return "Available functions for use:\n" + "\n".join([node.__repr__() for node in self.relevant_nodes]) + "If you intend to use this function, put the function calls into your generated function (assume the functions are already implemented and do not import them). Do not use it in a separate code block with your generated function. MAKE SURE THE FUNCTION IS USED AS A FUNCTION CALL SO IT CAN BE DETECTED BY AST\n"
        
    @property
    def referrable_function_dict(self):
        referrable_function_dict = {node.meta_prompt.func_name: node.code for node in self.relevant_nodes} # name to code of referrable functions 
        return referrable_function_dict
    
    
    def context_str(self, relevant_nodes: Optional[List['EvolNode']] = None):
        """ 
        Add sub-node description in context for parent node re-write
        - Use sub-nodes to add sub-functions for current node
        - Format context for parent node re-write
        """
        raise NotImplementedError
    
    
    def __repr__(self):
        desc_str = self.meta_prompt._desc_prompt()
        algorithm_str = f"Intuition: {self.reasoning}"
        quality_str = f"Fitness: {self.fitness:.2f}"
        return desc_str + "\n" + algorithm_str + "\n" + quality_str
    
    def __str__(self): 
        desc_str = self.meta_prompt._desc_prompt()
        algorithm_str = f"Intuition: {self.reasoning}"
        return desc_str + "\n" + algorithm_str



class QueryEngine:
    """
    QueryEngine is used to query the meta prompts from the library
    """
    def __init__(self, library_dir: str = "methods/nodes/", ignore_self: bool = False, self_func_name: str = None):
        self.library_dir = library_dir
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.ignore_self = ignore_self
        self.self_func_name = self_func_name
        self.meta_prompts = self.load_meta_prompts()
        

    def load_meta_prompts(self):
        meta_prompts = []
        for node_file in os.listdir(self.library_dir):
            if node_file.endswith("_node.json"):
                file_path = os.path.join(self.library_dir, node_file)
                with open(file_path, 'r') as f:
                    node_data = json.load(f)
                meta_prompt = MetaPrompt.from_dict(node_data['meta_prompt'])
                if self.ignore_self and meta_prompt.func_name == self.self_func_name:
                    continue
                meta_prompts.append(meta_prompt)
        return meta_prompts

    def _query_meta_prompt(self, task: str, top_k: int = 5) -> List[MetaPrompt]:
        """
        Query node json from library path and return top-k nodes
        """
        # Encode the task query
        query_embedding = self.sentence_transformer.encode(task)

        # Compute similarities between the query and all meta prompts
        similarities = []
        for meta_prompt in self.meta_prompts:
            prompt_embedding = self.sentence_transformer.encode(meta_prompt.task)
            similarity = cosine_similarity([query_embedding], [prompt_embedding])[0][0]
            similarities.append((meta_prompt, similarity))

        # Sort by similarity and get top-k results
        top_k_results = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

        # Return the top-k meta prompts as dictionaries
        return [meta_prompt for meta_prompt, _ in top_k_results]
    
    def query_node(self, task: str, top_k: int = 5) -> List['EvolNode']:
        meta_prompts = self._query_meta_prompt(task, top_k)
        return [EvolNode.load(meta_prompt.func_name) for meta_prompt in meta_prompts]
    


class PlanNode: 
    
    def __init__(self, meta_prompt: MetaPlan, 
                 get_response: Optional[Callable] = get_openai_response,
                 nodes: List[EvolNode] = [],
                 plan_dict: dict = {}):
        """ 
        Planning Node for subtask decomposition
        - Spawn helper nodes for better task performance
        """
        self.meta_prompt = meta_prompt 
        self._get_response = get_response 
        self.nodes = nodes
        self.relevant_nodes = None
        self.max_attempts = MAX_ATTEMPTS
        self.plan_dict = plan_dict
        self.plan_dicts = []

    def get_response(self, prompt: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(prompt, str):
            prompts = [prompt]
            n_prompt = 1
        else:
            prompts = prompt 
            n_prompt = len(prompts)
        try: 
            responses = self._get_response(prompts)
        except: 
            responses = []
            for p in prompts:
                responses.append(self._get_response(p))
        if n_prompt == 1:
            return responses[0]
        else:
            return responses      
    
    def _evolve_plan_dict(self, feedback: str = "", replace: bool = True, method: str = "i1", parents: list = []):
        
        err_msg = ""
        
        # Step 1: Generate Pseudo-Code for reliable sub-tasks decomposition
        prompt = getattr(self.meta_prompt, f"_get_prompt_{method}")(feedback, parents)

        self.query_nodes(ignore_self=replace, self_func_name=self.meta_prompt.func_name)
        prompt += "\n" + self.relevant_node_desc
        
        response = self.get_response(prompt) 
        code = extract_python_code(response)
        
        if code == "":
            err_msg += "No pseudo-code block found in the planning response: \n {response}\n"

        # Step 2: Generate Planning DAG: Multiple Nodes 
        graph_prompt = self.meta_prompt._get_plan_graph_prompt(code)
        plan_response = self.get_response(graph_prompt)
        try:
            plan_dict = extract_json_from_text(plan_response)
        except ValueError as e:
            plan_dict = {}
            err_msg += f"Failed to extract JSON from planning response:\n{e}\nResponse was:\n{plan_response}\n"
        
        plan_dict = self._update_plan_dict(plan_dict)
        plan_dict, err_msg_delta = check_n_rectify_plan_dict(plan_dict, self.meta_prompt)
        if err_msg_delta:
            err_msg += err_msg_delta
        
        return plan_dict, err_msg
    
    def evolve_plan_dict_sequential(self, feedback: str = "", method: str = "i1"): # Deprecated
        err_msg = ""
        for i in range(self.max_attempts):
            plan_dict, err_msg_delta = self._evolve_plan_dict(feedback, method=method)
            if err_msg_delta == "":
                self.plan_dict = plan_dict
                return plan_dict, ""
            err_msg += f"Plan evolution {i} failed with error message: \n{err_msg_delta}\n"
            
        return {}, err_msg
    
    def update_plan_dict(self, plan_dicts: list): 
        """ 
        TBD: Better, more sohpisticated approach required (explore all generate plans ... )
        """
        if len(plan_dicts) > 0:
            self.plan_dicts += plan_dicts
        else:
            self.plan_dicts = plan_dicts

        # Makeshift plan dict selection heuristic
        plan_dict = min(self.plan_dicts, key=lambda x: len(x.get("nodes", [])))
        self.plan_dict = plan_dict
    
    def evolve_plan_dict(self, feedback: str = "", method: str = "i1", parents: list = [], replace: bool = True, batch_size: int = 10):
        """ 
        Handling batch inference or sequential inference
        """
        print(f" :: Evolving {batch_size} plans in parallel...")


        err_msg = ""
        prompts = []
        prompt = getattr(self.meta_prompt, f"_get_prompt_{method}")(feedback, parents)
        self.query_nodes(ignore_self=replace, self_func_name=self.meta_prompt.func_name)
        prompt += "\n" + self.relevant_node_desc
        prompts = [prompt] * batch_size
        
        responses = self.get_response(prompts) # get pseudo-code for each plan
        print(" :: Pseudo-code generated for each plan")
        plan_dicts = []
        graph_prompts = []
        for response in responses:
            try:
                code = extract_python_code(response)
            except:
                code = ""
            
            if code == "":
                err_msg += "No pseudo-code block found in the planning response: \n {response}\n"

            # Step 2: Generate Planning DAG: Multiple Nodes 
            graph_prompt = self.meta_prompt._get_plan_graph_prompt(code)
            graph_prompts.append(graph_prompt)
            
        plan_responses = self.get_response(graph_prompts) # get plan_dict for each plan
        print(" :: Plan_dict generated for each plan")
        for plan_response in plan_responses:
            try:
                plan_dict = extract_json_from_text(plan_response)
            except ValueError as e:
                plan_dict = {}
                err_msg += f"Failed to extract JSON from planning response:\n{e}\nResponse was:\n{plan_response}\n"
            
            plan_dict = self._update_plan_dict(plan_dict)
            plan_dict, err_msg_delta = check_n_rectify_plan_dict(plan_dict, self.meta_prompt)

            if err_msg_delta:
                err_msg += err_msg_delta
            if plan_dict:
                plan_dicts.append(plan_dict)
                
        self.update_plan_dict(plan_dicts)
            
        return plan_dicts, err_msg
    
    
    def spawn_test_cases_majority(self, main_test_cases: list) -> tuple [bool, str]:
        def convert_unhash_to_hash(data):
            if isinstance(data, list):
                return tuple(convert_unhash_to_hash(item) for item in data)
            elif isinstance(data, dict):
                return frozenset((key, convert_unhash_to_hash(value)) for key, value in data.items())
            else:
                return data
            
        def convert_hash_to_unhash(data):
            if isinstance(data, tuple):
                return [convert_hash_to_unhash(item) for item in data]
            elif isinstance(data, frozenset):
                return {key: convert_hash_to_unhash(value) for key, value in data}
            else:
                return data
            
        def most_common(list_of_dicts):
            list_of_dicts = [convert_unhash_to_hash(d) for d in list_of_dicts]
            counter = Counter(list_of_dicts)
            sorted_dicts_by_frequency = sorted(counter.items(), key=lambda x: x[1], reverse=True)
            sorted_unique_dicts = [convert_hash_to_unhash(d) for d, _ in sorted_dicts_by_frequency]
            # pick the highest-count/earliest item
            return sorted_unique_dicts
                
        test_cases_dict, err_msg = spawn_test_cases(self.plan_dict, main_test_cases, get_multiple_response, unique=False)
        if test_cases_dict:
            for name, io in test_cases_dict.items():
                inputs, outputs = zip(*io)
                sub_list_input = most_common(inputs)[:len(main_test_cases)]
                sub_list_output = most_common(outputs)[:len(main_test_cases)]
                test_cases_dict[name] = list(zip(sub_list_input, sub_list_output))
            self.test_cases_dict = test_cases_dict
            print(f"Spawned {len(test_cases_dict)} test cases for all sub-nodes")
            return True, err_msg
        else:
            print(f"Failed to spawn test cases: {err_msg}")
            return False, err_msg 
    
    def spawn_test_cases_sequential(self, main_test_cases: list) -> tuple[bool, str]: # Deprecated
        test_cases_dict, err_msg = spawn_test_cases(self.plan_dict, main_test_cases, self.get_response, SPAWN_TEST_MAX_TRIES)
        if test_cases_dict:
            self.test_cases_dict = test_cases_dict
            print(f"Spawned {len(test_cases_dict)} test cases for all sub-nodes")
            return True, err_msg
        else:
            print(f"Failed to spawn test cases: {err_msg}")
            return False, err_msg 
        
    def spawn_test_cases(self, main_test_cases: list, batch_size: int = 1) -> tuple[bool, str]:
        test_cases_dict, err_msg = spawn_test_cases(self.plan_dict, main_test_cases, self.get_response, SPAWN_TEST_MAX_TRIES)
        if test_cases_dict:
            self.test_cases_dict = test_cases_dict
            print(f"Spawned {len(test_cases_dict)} test cases for all sub-nodes")
            return True, err_msg
        else:
            print(f"Failed to spawn test cases: {err_msg}")
            return False, err_msg 
    
    def evolve_sub_nodes(self, method: str = "i1", batch_size: int = NODE_EVOLVE_BATCH_SIZE):
        """
        Evolve all sub-nodes using their respective test cases
        """
        for i, node_dict in enumerate(self.plan_dict["nodes"]):
            meta_prompt = MetaPrompt(
                task=node_dict["task"],
                func_name=node_dict["name"],
                inputs=node_dict["inputs"],
                outputs=node_dict["outputs"],
                input_types=node_dict["input_types"],
                output_types=node_dict["output_types"],
                mode=PromptMode((node_dict.get("mode", "code")).lower())
            )
            test_cases = self.test_cases_dict[node_dict["name"]]
            if "fitness" in node_dict and "code" in node_dict: 
                node = EvolNode(meta_prompt, node_dict["code"], node_dict["reasoning"], get_response=self._get_response, test_cases=test_cases, fitness=node_dict["fitness"])
            else:
                node = EvolNode(meta_prompt, None, None, get_response=self._get_response, test_cases=test_cases)
                print(f"🎲 :: Evolving {node.meta_prompt.func_name} ... ({i+1}/{len(self.plan_dict['nodes'])})")
                node.evolve(method, replace=True, max_tries=2, num_runs=2, batch_size=batch_size) # It's funny how 30+ sec could elapse before llm inference ... (collecting prompts ?? wtf is taking so long ??)
            self.nodes.append(node)
            
            
    def evolve_plan(self, method: str = "i1"):
        """ 
        Allocate budget to each plans, ensemble fitness of nodes after budget exhausted and re-allocate more budgets to a selection of plans
        """
        raise NotImplementedError
    
    def _spawn_nodes(self, plan_dict: Dict): # bad idea since test cases are not grounded ... 
        """ 
        Spawn new nodes based on plan_dict
        - Each node is evolved as CODE and PROMPT, we let evaluation result decides which one to keep
        """
        prompt_nodes = {}
        code_nodes = {}
        for node in plan_dict["nodes"]:
            meta_prompt_node = MetaPrompt(
                task=node.get("task"),
                func_name=node.get("name"),
                inputs=node.get("inputs"),
                outputs=node.get("outputs"),
                input_types=node.get("input_types"),
                output_types=node.get("output_types"),
                mode = PromptMode.PROMPT
            )
            prompt_nodes[node.get("name")] = EvolNode(meta_prompt=meta_prompt_node)
                
            meta_code_node = MetaPrompt(
                task=node.get("task"),
                func_name=node.get("name"),
                inputs=node.get("inputs"),
                outputs=node.get("outputs"),
                input_types=node.get("input_types"),
                output_types=node.get("output_types"),
                mode = PromptMode.CODE
            )
            code_nodes[node.get("name")] = EvolNode(meta_prompt=meta_code_node)

        return prompt_nodes, code_nodes
    
    def evolve_node(self, node_id: str, method: str):
        """ 
        Evolve each node (pick the best between prompt and code ver.)
        """
        raise NotImplementedError
    
    def save(self, library_dir: str = "methods/plans/") -> None:
        """ 
        Save plan_dict, as well as sub-nodes (if there is any ...)
        """
        # Create the directory if it doesn't exist
        os.makedirs(library_dir, exist_ok=True)

        # Save the plan details
        plan_data = {
            "meta_prompt": self.meta_prompt.to_dict(),  # Assuming MetaPlan has a to_dict method
            "plan_dict": self._evolve_plan_dict()
        }

        plan_path = os.path.join(library_dir, f"{self.meta_prompt.func_name}_plan.json")
        with open(plan_path, 'w') as f:
            json.dump(plan_data, f, indent=2)

        # Save sub-nodes if they exist
        if self.nodes:
            nodes_dir = os.path.join(library_dir, self.meta_prompt.func_name)
            os.makedirs(nodes_dir, exist_ok=True)
            for node in self.nodes:
                node.save(nodes_dir)

    @classmethod 
    def load(cls, plan_name: str, plan_dir: str = "methods/plans/", get_response: Optional[Callable] = get_openai_response) -> 'PlanNode':
        plan_path = os.path.join(plan_dir, f"{plan_name}_plan.json")
        with open(plan_path, 'r') as f:
            plan_data = json.load(f)

        meta_plan = MetaPlan.from_dict(plan_data['meta_prompt'])  # Assuming MetaPlan has a from_dict method
        plan_node = cls(meta_prompt=meta_plan)
        plan_node.plan_dict = plan_data['plan_dict']

        # Load sub-nodes if they exist
        nodes_dir = os.path.join(plan_dir, plan_name)
        if os.path.exists(nodes_dir):
            plan_node.nodes = []
            for node_file in os.listdir(nodes_dir):
                if node_file.endswith('_node.json'):
                    node_name = node_file[:-10]  # Remove '_node.json'
                    node = EvolNode.load(node_name, nodes_dir, get_response)
                    plan_node.nodes.append(node)

        return plan_node
    
    
    def query_nodes(self, top_k: int = 5, ignore_self: bool = False, self_func_name: str = None) -> List['EvolNode']:
        """ 
        Query nodes from library
        """
        query_engine = QueryEngine(ignore_self=ignore_self, self_func_name=self_func_name)
        self.relevant_nodes = query_engine.query_node(self.meta_prompt.task)
        
    def _update_plan_dict(self, plan_dict):
        if plan_dict == {}:
            return {}
        for node in self.relevant_nodes:
            for sub_node in plan_dict.get("nodes", []):
                if node.meta_prompt.func_name == sub_node["name"]:
                    for k in ["inputs", "input_types", "outputs", "output_types"]:
                        sub_node[k] = getattr(node.meta_prompt, k)
                    for k in ["code", "reasoning", "fitness"]:
                        sub_node[k] = getattr(node, k)
        return plan_dict
        
    @property 
    def relevant_node_desc(self):
        if len(self.relevant_nodes) == 0:
            return ""
        return "You could call these available functions without defining them:\n" + "\n".join([str(node) for node in self.relevant_nodes])
        
    @property
    def referrable_function_dict(self):
        referrable_function_dict = {node.meta_prompt.func_name: node.code for node in self.relevant_nodes} # name to code of referrable functions 
        return referrable_function_dict
    
