from abc import ABC, abstractmethod
from .meta_prompt import MetaPrompt, PromptMode, parse_evol_response
from .meta_prompt import MetaPlan, extract_json_from_text, ALIGNMENT_CHECK_PROMPT
from .meta_execute import call_func_code, call_func_prompt
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


def check_alignment(pred_output: dict, target_output: dict, get_response: Optional[Callable] = get_openai_response, max_tries: int = 3):
    """ 
    Alignment checking with expected outputs with LLM
    - we have two dictionary, need to make sure they are aligned
    - 1. they have same keys 
    - 2. their values are 'basically the same'
    """
    
    prompt = ALIGNMENT_CHECK_PROMPT.format(pred_output=pred_output, target_output=target_output)
    for i in range(max_tries):
        try:
            response = get_response(prompt)
            check_dict = extract_json_from_text(response)
            if check_dict['aligned']:
                return True
        except Exception as e:
            print(f"Alignment check failed on try {i+1}, retrying...")
    return False

#################################
#   Evolution on Graph          #
#   - Node: Code, Prompt, Tool  #
#   - Edge: Not Decided Yet     #
#################################


# Evolution: 
# - (Hyper-parameter) Meta-Heuristic: Task, FuncName, Input, Output
# - (get_prompt_xx) Evolution tactic specific prompt templating, used for generating node
# - (_get_node) Get prompt response from LLM, parse node content (code, tool, prompt) 
# - (xx) i1/e1/m1/m2/e2 Evolution search on node


class EvolNode:
    
    def __init__(self, meta_prompt: MetaPrompt, code: Optional[str] = None, reasoning: Optional[str] = None,
                 get_response: Optional[Callable] = get_openai_response):
        """ 
        Executable Task
        """
        self.code = code
        self.reasoning = reasoning
        self.fitness = 0.0
        self.meta_prompt = meta_prompt
        self.test_cases = []
        self.get_response = get_response
        
        
    def _get_extend_test_cases_response(self, num_cases: int = 5, feedback: str = ""):
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


    def _evolve(self, method: str, parents: list = None, replace=False, error_msg: str = ""):
        """
        Note: Evolution process will be decoupled with the fitness assignment process
        """
        prompt_method = getattr(self.meta_prompt, f'_get_prompt_{method}')
        prompt_content = prompt_method()
        prompt_content += error_msg # Append error message to the prompt
     
        response = self.get_response(prompt_content)
        reasoning, code = parse_evol_response(response)
        
        if replace:
            self.reasoning, self.code = reasoning, code
        return reasoning, code
    
    def evolve(self, method: str, parents: list = None, replace=False, max_attempts: int = 5, fitness_threshold: float = 0.8, num_runs: int = 5):
        """
        Evolve node and only accept structurally fit solutions
        Attempts multiple evolutions before returning the final output
        """
        
        
        for attempt in range(max_attempts):
            reasoning, code = self._evolve(method, parents, replace=False)    
            _, fitness, error_msg = self._evaluate_fitness(code=code, max_tries=1, num_runs=num_runs)            
            
            if fitness >= self.fitness:
                if replace:
                    self.reasoning, self.code = reasoning, code
            if fitness >= fitness_threshold:
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
                try:
                    output_value = call_func_code(test_input, code, self.meta_prompt.func_name, file_path=None)
                    passed_tests += 1
                except Exception as e:
                    error_msg += str(e)
            elif self.meta_prompt.mode == PromptMode.PROMPT:
                try:
                    output_dict = call_func_prompt(test_input, code, self.get_response)
                    [output_dict.get(output_name) for output_name in self.meta_prompt.outputs] # check for all outputs
                    passed_tests += 1
                except Exception as e:
                    error_msg += str(e)
            else:
                raise ValueError(f"Unknown mode: {self.meta_prompt.mode}")

        return passed_tests / total_tests, error_msg
    
    def call_prompt_func(self, test_input: Dict, code: Optional[str] = None, max_tries: int = 3):
        if code is None:
            code = self.code
        for i in range(max_tries):
            try:
                output_dict = call_func_prompt(test_input, code, self.get_response)
                return output_dict
            except Exception as e:
                continue 
        raise ValueError(f"Failed to get dictionary output after {max_tries} attempts")
    
    def call_code_func(self, test_input: Dict, code: Optional[str] = None, file_path: Optional[str] = None):
        if code is None:
            code = self.code
        try:
            output_value = call_func_code(test_input, code, self.meta_prompt.func_name, file_path=file_path)
            output_name = self.meta_prompt.outputs[0]
            output_dict = {output_name: output_value}
            return output_dict
        except Exception as e:
            raise ValueError(f"Failed to get output value: {e}")
        
    
    def _evaluate_fitness(self, test_cases: Optional[List[Tuple[Dict, Dict]]] = None, code: Optional[str] = None, 
                                        max_tries: int = 3, num_runs: int = 1) -> float:
        """ 
        Alignment checking with expected outputs with LLM
        """
        if test_cases is None:
            test_cases = self.test_cases
            
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
                try:
                    output_dict = self.call_code_func(test_input, code, file_path=None)
                    compiled_tests += 1
                    is_aligned = check_alignment(output_dict, test_output, self.get_response)
                    if is_aligned:
                        passed_tests += 1
                    if not is_aligned:
                        issue_summary += f"Input: {test_input}, Pred: {output_dict}, Expected: {test_output}\n"
                except Exception as e:
                    issue_summary += f"Given Input: {test_input}. Can't parse output dictionary from LLM's response.\n"
                    error_msg += str(e)
                    
            elif self.meta_prompt.mode == PromptMode.PROMPT:
                try:
                    output_dict = self.call_prompt_func(test_input, code, max_tries)
                    [output_dict.get(output_name) for output_name in self.meta_prompt.outputs] # check for all outputs
                    compiled_tests += 1
                    is_aligned = check_alignment(output_dict, test_output, self.get_response)
                    if is_aligned:
                        passed_tests += 1
                    if not is_aligned:
                        issue_summary += f"Input: {test_input}, Pred: {output_dict}, Expected: {test_output}\n"
                except Exception as e:
                    issue_summary += f"Given Input: {test_input}. Can't parse output dictionary from LLM's response.\n"
                    error_msg += str(e)
            else:
                raise ValueError(f"Unknown mode: {self.meta_prompt.mode}")
            
        # add overal information
        global_summary = f"--- Compiled {compiled_tests} out of {total_tests} test cases\n"
        global_summary += f"--- Passed {passed_tests} out of {total_tests} test cases\n"
        issue_summary = global_summary + issue_summary

        structural_fitness = compiled_tests / total_tests
        functional_fitness = passed_tests / total_tests
        
        return structural_fitness, functional_fitness, issue_summary


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
            output_value = call_func_code(inputs, self.code, self.meta_prompt.func_name, file_path=None) # TODO: extend to multiple outputs ...
            output_name = self.meta_prompt.outputs[0]
            return {output_name: output_value}
        elif self.meta_prompt.mode == PromptMode.PROMPT:
            output_name = self.meta_prompt.outputs[0]
            
            errors = []
            for attempt in range(max_attempts):
                try:
                    output_dict = call_func_prompt(inputs, self.code, self.get_response)
                    output_value = output_dict.get(output_name, None)  # We don't like surprises
                    if output_value is None:
                        raise ValueError(f"Output value for {output_name} is None")
                    return {output_name: output_value}
                except Exception as e:
                    errors.append(str(e))
                    if attempt == max_attempts - 1:
                        error_msg = "; ".join(errors)
                        raise ValueError(f"Failed to get output after {max_attempts} attempts: {error_msg}")
        
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
    def load(cls, node_name: str, library_dir: str = "methods/nodes/") -> 'EvolNode':
        node_path = os.path.join(library_dir, f"{node_name}_node.json")
        with open(node_path, 'r') as f:
            node_data = json.load(f)
        meta_prompt = MetaPrompt.from_dict(node_data['meta_prompt'])  # Assuming MetaPrompt has a from_dict method
        node = cls(meta_prompt=meta_prompt, code=node_data['code'], reasoning=node_data['reasoning'])
        node.test_cases = [tuple([test_case['input'], test_case['expected_output']]) for test_case in node_data['test_cases']]
        return node

    

class EvolGraph:
    
    def __init__(self, nodes: Optional[List[EvolNode]], edges: Optional[List], get_response: Optional[Callable] = get_openai_response):
        """ 
        EvolGraph: Topology of Node functions -- Plan. 
        """
        self.nodes: Dict[str, EvolNode] = nodes
        self.edges = edges
        self.get_response = get_response
    
    @classmethod
    def generate(cls, goal: str, get_response: Optional[Callable] = get_openai_response):
        """ 
        DNA -> RNA -> Protein ....
        - Goal -> MetaPlan -> PlanGraph
        - PlanGraph -> MetaPrompt -> EvolNode
        """
        meta_prompt = MetaPlan(goal)
        prompt_content = meta_prompt._get_prompt_i1()
        response = get_response(prompt_content)
        plan_dict = extract_json_from_text(response)
        
        nodes = {}
        for node in plan_dict["nodes"]:
            node_prompt = MetaPrompt(
                task=node.get("task"),
                func_name=node.get("name"),
                inputs=node.get("inputs"),
                outputs=node.get("outputs"),
                input_types=node.get("input_types"),
                output_types=node.get("output_types"),
                mode=node.get("mode").lower()
            )
            nodes[node.get("name")] = EvolNode(meta_prompt=node_prompt)
        
        edges = plan_dict["edges"]
        
        prompt_content = meta_prompt._get_eval_prompt_i1()
        response = get_response(prompt_content)
        plan_dict = extract_json_from_text(response)
        eval_prompt = MetaPrompt(
            task=node.get("task"),
            func_name=node.get("name"),
            input=node.get("input"),
            output=node.get("output"),
            mode=node.get("mode").lower()
        )
        eval_node = EvolNode(meta_prompt=eval_prompt)
        
        graph = cls(nodes=list(nodes.values()), edges=edges, get_response=get_response)

        return graph
    
    def evolve_graph(self, method: str):
        """ 
        Evolve Planning Graph Topology
        """ 
        raise NotImplementedError

    def evolve_node(self, node_id: str, method: str):
        node = self.get_node(node_id)
        if node:
            input_nodes = self.get_input_nodes(node_id)
            if method == 'i1':
                return node.i1()
            elif method in ['e1', 'e2', 'm1', 'm2']:
                return getattr(node, method)(input_nodes)
        return None
    
    def save(self, graph_path: str) -> None:
        raise NotImplementedError # if you are not tested, you are not implemented
        graph_data = {
            "nodes": {name: node.save(os.path.join(graph_path, f"node_{name}.json")) 
                      for name, node in self.nodes.items()},
            "edges": self.edges,
            "eval_node": self.eval_node.save(os.path.join(graph_path, "eval_node.json")) if self.eval_node else None
        }
        os.makedirs(graph_path, exist_ok=True)
        with open(os.path.join(graph_path, "graph_structure.json"), 'w') as f:
            json.dump(graph_data, f, indent=2)

    @classmethod
    def load(cls, graph_path: str) -> 'EvolGraph':
        raise NotImplementedError
        with open(os.path.join(graph_path, "graph_structure.json"), 'r') as f:
            graph_data = json.load(f)
        
        nodes = {name: EvolNode.load(os.path.join(graph_path, f"node_{name}.json")) 
                 for name in graph_data["nodes"]}
        edges = graph_data["edges"]
        eval_node = EvolNode.load(os.path.join(graph_path, "eval_node.json")) if graph_data["eval_node"] else None
        
        return cls(nodes=nodes, edges=edges, eval_node=eval_node)
    
    @classmethod
    def read_from_dict(cls, plan_dict: Dict):
        """
        Create an EvolGraph instance from a dictionary representation.
        
        :param plan_dict: A dictionary containing 'nodes' and 'edges' keys.
        :return: An EvolGraph instance.
        """
        nodes = {}
        for node_data in plan_dict.get("nodes", []):
            node_prompt = MetaPrompt(
                task=node_data.get("task"),
                func_name=node_data.get("name"),
                inputs=[node_data.get("input")],  # Wrap in list as MetaPrompt expects a list
                outputs=[node_data.get("output")],  # Wrap in list as MetaPrompt expects a list
                input_types=node_data.get("input_types", []),  # Assuming string input type, adjust if needed
                output_types=node_data.get("output_types", []),  # Assuming string output type, adjust if needed
                mode=PromptMode.PROMPT if node_data.get("mode", "").lower() == "prompt" else PromptMode.CODE
            )
            nodes[node_data.get("name")] = EvolNode(meta_prompt=node_prompt)
        
        edges = plan_dict.get("edges", [])
        
        # For now, we'll set eval_node to None. You may want to add logic to create it if needed.
        eval_node = None

        return cls(nodes=nodes, edges=edges, eval_node=eval_node)
