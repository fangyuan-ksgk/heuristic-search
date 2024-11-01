from abc import ABC, abstractmethod
from .meta_prompt import MetaPrompt, PromptMode, parse_evol_response
from .meta_prompt import MetaPlan, extract_json_from_text, extract_python_code, ALIGNMENT_CHECK_PROMPT
from .meta_execute import call_func_code, call_func_prompt, compile_code_with_references
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
from .llm import get_openai_response
import re, os, json
from tqdm import tqdm 
from collections import defaultdict
from typing import Optional, Dict, List, Callable, Tuple


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
        return line.split("//")[0].split("#")[0] # Remove comments 
    return ('\n').join(map(clean_line, s.split('\n')))


def require_llm_metric(value) -> bool:
    if isinstance(value, int): 
        return False
    elif isinstance(value, float):
        return False
    else:
        return True

def type_to_metric(value) -> Callable:
    
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
        return lambda x, y: True, ""


def _check_alignment_with_metric(pred_output: dict, target_output: dict):
    """ 
    Input prediction & target dictionary, design algorithm for specific output value types' matching
    """
    error_msg = ""
    for key, target_value in target_output.items():
        if key not in pred_output:
            error_msg += f"Key {key} not found in prediction output\n"
            return False, error_msg
        
        pred_value = pred_output[key]
        metric = type_to_metric(target_value)
        
        is_aligned, error_msg_delta = metric(pred_value, target_value)
        if not is_aligned:
            error_msg += error_msg_delta + "\n"
            error_msg += f"Value mismatch for key {key}: {pred_value} != {target_value}\n"
            return False, error_msg

    return True, ""


def _check_alignment_with_llm(pred_output: dict, target_output: dict, get_response: Optional[Callable] = get_openai_response, max_tries: int = 3):
    """ 
    Alignment checking with expected outputs with LLM
    """
    error_msg = ""
    prompt = ALIGNMENT_CHECK_PROMPT.format(pred_output=pred_output, target_output=target_output)

    trimmed_pred_output = {k: v for k, v in pred_output.items() if require_llm_metric(v)}
    trimmed_target_output = {k: v for k, v in target_output.items() if require_llm_metric(v)}
    
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

def check_alignment(pred_output: dict, target_output: dict, get_response: Optional[Callable] = get_openai_response, max_tries: int = 3):
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
    _, error_msg_delta = _check_alignment_with_llm(pred_output, target_output, get_response, max_tries)
    error_msg += error_msg_delta
    
    if error_msg == "":
        return True, ""
    else:
        return False, error_msg
    
    
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
                 fitness: float = 0.0):
        """ 
        Executable Task
        """
        self.code = code
        self.reasoning = reasoning
        self.fitness = fitness
        self.meta_prompt = meta_prompt
        self.test_cases = []
        self.get_response = get_response
        self.relevant_nodes = []
        self.error_msg = "" # contains information about encountered error :: TBD :: use LLM to summarize it
        
        if test_cases is not None:
            self.test_cases = test_cases
        else:
            self.get_test_cases(3) # generate 3 test cases for new node
            
        
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
        except:
            pass # alas, response parsing failed
        
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
    
    
    def _get_evolve_response(self, method: str, parents: Optional[list] = None, feedback: str = ""):
        prompt_method = getattr(self.meta_prompt, f'_get_prompt_{method}')
        prompt_content = prompt_method(parents)
        prompt_content += self.relevant_node_desc
        prompt_content += "\nIdea: " + feedback # External Guidance (perhaps we should reddit / stackoverflow this thingy)
     
        response = self.get_response(prompt_content)
        return response 

    def _evolve(self, method: str, parents: list = None, replace=False, feedback: str = ""):
        """
        Note: Evolution process will be decoupled with the fitness assignment process
        """
        response = self._get_evolve_response(method, parents, feedback)
        
        try:
            reasoning, code = parse_evol_response(response)
            code = compile_code_with_references(code, self.referrable_function_dict) # deal with node references
        except Exception as e:
            print("Parse Response Failed...")
            return None, None 
        
        if replace:
            self.reasoning, self.code = reasoning, code
        return reasoning, code
    
    def evolve(self, method: str, parents: list = None, replace=False, feedback: str = "", max_attempts: int = 5, fitness_threshold: float = 0.8, num_runs: int = 5):
        """
        Evolve node and only accept structurally fit solutions
        Attempts multiple evolutions before returning the final output
        """
        
        # Query once 
        self.query_nodes(ignore_self=replace, self_func_name=self.meta_prompt.func_name)
        
        # Evolve many times
        for attempt in range(max_attempts):
            reasoning, code = self._evolve(method, parents, replace=False) 
            self.tmp_code = code   
            fitness, error_msg = self._evaluate_fitness(code=code, max_tries=1, num_runs=num_runs)            
            
            if replace and fitness >= self.fitness:
                
                print("--- Replacing with new node") 
                self.reasoning, self.code = reasoning, code # Always replace worse with better
                self.fitness = fitness
                self.error_msg = error_msg
                
                if fitness >= fitness_threshold: # Terminate evolution only when fitness threshold is reached
                    print(f"--- Fitness: {fitness:.2f}")
                    print("--- Fitness threshold reached")
                    return reasoning, code
            
            
            
            # If not successful, log the attempt
            print(f" - Attempt {attempt + 1} failed. Fitness: {fitness:.2f}. Error: {error_msg}")
        
        # If all attempts fail, return None
        print(f"Evolution failed after {max_attempts} attempts.")
        return None, None
    
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
    
    def call_prompt_function(self, test_input: Dict, code: Optional[str] = None, max_tries: int = 3):
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
    
    def call_code_function(self, test_input: Dict, code: Optional[str] = None, file_path: Optional[str] = None):
        if code is None:
            code = self.code
        
        output_value, error_msg = call_func_code(test_input, code, self.meta_prompt.func_name, file_path=file_path)
        output_name = self.meta_prompt.outputs[0]
        output_dict = {output_name: output_value}
        return output_dict, error_msg
        
    
    def _evaluate_fitness(self, test_cases: Optional[List[Tuple[Dict, Dict]]] = None, code: Optional[str] = None, 
                                        max_tries: int = 3, num_runs: int = 1) -> float:
        """ 
        Alignment checking with expected outputs with LLM
        """
        if code is None:
            return 0.0, ""
        
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
                    is_aligned, error_msg_delta2 = check_alignment(output_dict, test_output, self.get_response)
                    
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
                    is_aligned = check_alignment(output_dict, test_output, self.get_response)
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
        
        return fitness(), f" {str(fitness)}\n" + issue_summary + "\nError Message:\n" + error_msg


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
    
    def __call__(self, inputs, max_attempts: int = 3):
        """ 
        TBD: Inheritance to accumulated codebase with 'file_path' | Graph Topology naturally enables inheritance
        TBD: Stricter input / output type checking to ensure composibility
        """
        if self.meta_prompt.mode == PromptMode.CODE:
            output_value, err_msg = call_func_code(inputs, self.code, self.meta_prompt.func_name, file_path=None) # TODO: extend to multiple outputs ...
            output_name = self.meta_prompt.outputs[0]
            return {output_name: output_value} # assuming single output for code-based node
        
        elif self.meta_prompt.mode == PromptMode.PROMPT:
            output_name = self.meta_prompt.outputs[0]
            
            errors = []
            for _ in range(max_attempts):
                output_dict, err_msg = self.call_prompt_function(inputs, self.code, max_tries=1)
                if output_dict is None or output_name not in output_dict:
                    value_error_msg = f"Output value for {output_name} is None. Output dict: {output_dict} with error message: {err_msg}"
                    errors.append(value_error_msg)
                    continue            
                else:
                    return output_dict 
                
            error_str = "\n".join(errors)
            raise ValueError(error_str)
    
    def save(self, library_dir: str = "methods/nodes/") -> None:
        node_data = {
            "code": self.code,
            "reasoning": self.reasoning,
            "meta_prompt": self.meta_prompt.to_dict(),  # Assuming MetaPrompt has a to_dict method
            "test_cases": [{"input": test_case[0], "expected_output": test_case[1]} for test_case in self.test_cases]
        }
        node_path = os.path.join(library_dir, f"{self.meta_prompt.func_name}_node.json")
        os.makedirs(os.path.dirname(node_path), exist_ok=True)
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
                   get_response=get_response)
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
        return "Available functions for use:\n" + "\n".join([node.__repr__() for node in self.relevant_nodes]) + "If you intend to use this function, put the function calls into your generated function (assume the functions are already implemented). Do not use it in a separate code block with your generated function.\n"
        
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
                 nodes: Optional[List[EvolNode]] = None):
        """ 
        Planning Node for subtask decomposition
        - Spawn helper nodes for better task performance
        """
        self.meta_prompt = meta_prompt 
        self.get_response = get_response 
        self.nodes = nodes
        
    def _evolve_plan_dict(self):
        
        # Step 1: Generate Pseudo-Code for SubTask Decomposition
        prompt = self.meta_prompt._get_pseudo_code_prompt() # Pseudo-Code Prompt (Non-implemented functional)
        response = self.get_response(prompt) # Use Strong LLM to build up pseudo-code
        code = extract_python_code(response) # Extract Python Code from response 

        # Step 2: Generate Planning DAG: Multiple Nodes 
        graph_prompt = self.meta_prompt._get_plan_graph_prompt(code) 
        plan_response = self.get_response(graph_prompt)
        plan_dict = extract_json_from_text(plan_response)
        return plan_dict
    
    def _spawn_nodes(self, plan_dict: Dict):
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