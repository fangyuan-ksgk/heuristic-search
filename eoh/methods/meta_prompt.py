from dataclasses import dataclass
from enum import Enum
import re
import json
import re
import ast
import networkx as nx
from dataclasses import dataclass
from typing import Optional, Union, Callable
import test

class PromptMode(Enum):
    CODE = "code"
    TOOL = "tool"
    PROMPT = "prompt"
    
    
def get_prompt_mode(mode: str):
    if mode in [PromptMode.CODE.value, PromptMode.PROMPT.value]:
        return PromptMode(mode)
    if "CODE" in mode:
        return PromptMode.CODE
    else:
        return PromptMode.PROMPT
    
# MetaPrompt describe meta-heuristic for each node's generation (i1) evolution (e1, e2) and mutation (m1, m2)
# -- Meta Heuristic includes Task, Function Input and Output
# -- Prompt Templating Function for Node's evolution
# -- Covering 3 types of Node (Code, Prompt, Tool)
# -- Covering Generation (i1) Evolution (e1, e2) and Mutation (m1, m2) for each node


@dataclass
class MetaPrompt:
    task: str
    func_name: str
    inputs: list
    outputs: list
    input_types: list 
    output_types: list
    mode: PromptMode
    
    @property
    def joined_inputs(self):
        if len(self.inputs) > 1:
            return ", ".join("'" + s + "'" for s in self.inputs)
        else:
            return "'" + self.inputs[0] + "'"
        
    @property
    def joined_outputs(self):
        if len(self.outputs) > 1:
            return ", ".join("'" + s + "'" for s in self.outputs)
        else:
            return "'" + self.outputs[0] + "'"
        
    def _desc_prompt(self):
        if self.mode == PromptMode.CODE:
            prompt_content = f"Function name: {self.func_name}, this fuction accept {len(self.inputs)} input(s): {self.joined_inputs} with types {', '.join(self.input_types)}. "\
                f"The function return {len(self.outputs)} output(s): {self.joined_outputs} with types {', '.join(self.output_types)}. This function is used for {self.task}"
            return prompt_content
        elif self.mode == PromptMode.PROMPT:
            prompt_content = f"Function name: {self.func_name}, this fuction accept {len(self.inputs)} input(s): {self.joined_inputs}. "\
                f"The function uses a LLM to complete the task: {self.task}."
            return prompt_content
        
    def _base_prompt(self):
        if self.mode == PromptMode.CODE:
            prompt_content = f"First, describe your new algorithm and main steps in one sentence. "\
                "The description must be inside a brace. Next, implement it in Python as a function named "\
                f"{self.func_name}. This function should accept {len(self.inputs)} input(s): "\
                f"{self.joined_inputs} with types {', '.join(self.input_types)}. "\
                f"The function should return {len(self.outputs)} output(s): "\
                f"{self.joined_outputs} with types {', '.join(self.output_types)}."\
                "Make sure to include type hints in your function signature."
            return prompt_content
        elif self.mode == PromptMode.PROMPT:
            prompt_content = (
                f"First, describe your new reasoning and main thoughts in one sentence. "
                "The description must be inside a brace. Implement a Python function that generates a prompt to guide an AI in completing the task. "
                "Second, in another brace, evaluate all of the given tools and decide for each tool whether they are useful for the AI to complete the task. If you determine they are, use it."
                f"Then, code a function that completes the task. Follow these specifications: - Function name: generate_prompt - Input parameters: {self.joined_inputs} - Return value: A string containing the final prompt for the AI. "
                f"Ask for JSON-style response with output dictionary: {{" + ', '.join(f"'{out}': {type_hint}(...)" for out, type_hint in zip(self.outputs, self.output_types)) + "}} in a markdown format\n"
                "Make sure that you generate a prompt that is able to output the json-style response (in markdown) and maybe some reasoning. Tell the AI directly only these 2 items are needed or it will give a wrong response. You can do this by commanding it to follow a certain output structure with json, however you are not allowed to use backticks in your prompt. Make sure the output is in json format with right types or people will die\n"
                "Your function should incorporate the reasoning from step 1 and use the input parameters to create a tailored prompt for the task. Make sure to only give one code block, if more are given people will die. The AI is only able to access the returned prompt, and nothing else.\n\n"
                "IMPORTANT: Only you have access to the tools given to you, the AI is unable to use it, so if you find a tool very useful for the AI, use it and put its output in the prompt you give to the AI. DO not make the AI do work when you have a tool that already does that work and do not preprocess the output, use the AI to preprocess for you. DO not assume what tools you have, only use the tools given.\n"
                "Example of a prompt to the AI: 'Given the input data, {{input}}, do <task>. Make sure the output is a json string in markdown like this {{<key>:<value>}} near the top of your response or people will die.'"
            )
            return prompt_content
        elif self.mode == PromptMode.TOOL:
            raise NotImplementedError
        
    def _get_eval_prompt(self, num_cases: int = 5):
        """ 
        Asking for (input, output) pairs for evaluation
        """
        prompt_content = f"Task: {self.task}\n\n"\
                         f"Create 5 diverse test cases for the Python function '{self.func_name}'. Each test case should be an (input, output) pair. "\
                         "Include a variety of scenarios, especially edge cases, to thoroughly test the function.\n\n"\
                         "Format your response as a list of dictionaries in JSON format. Each dictionary should contain 'input' and 'expected_output' keys. "\
                         "Make sure the data types in your test cases match the function's input and output types.\n\n"\
                         "Example format:\n"\
                         "```json\n"\
                         "[\n"\
                         "    {\n"\
                         "        'input': {" + ', '.join(f"'{inp}': {type_hint}(...)" for inp, type_hint in zip(self.inputs, self.input_types)) + "},\n"\
                         "        'expected_output': {" + ', '.join(f"'{out}': {type_hint}(...)" for out, type_hint in zip(self.outputs, self.output_types)) + "}\n"\
                         "    },\n"\
                         "    ...\n"\
                         "]\n"\
                         "```\n"\
                         f"Provide {num_cases} such pairs, ensuring type correctness and diversity in the inputs and outputs."        
        return prompt_content
    
    def _get_eval_prompt_with_feedback(self, num_cases: int = 5, feedback: str = ""):
        prompt_content = self._get_eval_prompt(num_cases)
        if feedback != "":
            prompt_content += f"\nCreate evaluation pairs focus on incorporating previous feedback: {feedback}"
        return prompt_content
    
    def _get_prompt_indivs(self, indivs: Union[list, dict]) -> str:
        if isinstance(indivs, dict):
            indivs = [indivs]
            
        prompt_indiv = ""
        for i, indiv in enumerate(indivs, 1):
            if self.mode == PromptMode.CODE:
                prompt_indiv += f"No.{i}:\n[ALGORITHM]: {indiv['reasoning']}\n[CODE]: {indiv['code']}\n"
            elif self.mode == PromptMode.PROMPT:
                prompt_indiv += f"No.{i}:\n[APPROACH]: {indiv['reasoning']}\n[PROMPT FUNCTION]: {indiv['code']}\n"                    
        return prompt_indiv
        
    def _get_prompt_i1(self, indivs: Optional[list] = None):
        prompt_content = f"{self.task}\n{self._base_prompt()}"
        return prompt_content
        
    def _get_prompt_e1(self, indivs: list):
        if self.mode == PromptMode.CODE:
            prompt_indiv = self._get_prompt_indivs(indivs)

            prompt_content = f"{self.task}\n"\
                f"I have {len(indivs)} existing algorithms with their codes as follows:\n"\
                f"{prompt_indiv}"\
                "Please help me create a new algorithm that has a totally different form from the given ones.\n"\
                f"{self._base_prompt()}"
            return prompt_content
        elif self.mode == PromptMode.PROMPT:
            prompt_indiv = self._get_prompt_indivs(indivs)

            prompt_content = f"{self.task}\n"\
                f"I have {len(indivs)} existing prompt generation approaches with their functions as follows:\n"\
                f"{prompt_indiv}"\
                "Please help me create a new prompt generation approach that is totally different from the given ones.\n"\
                f"{self._base_prompt()}"
            return prompt_content
        
    def _get_prompt_e2(self, indivs: list):
        if self.mode == PromptMode.CODE:
            prompt_indiv = self._get_prompt_indivs(indivs)

            prompt_content = f"{self.task}\n"\
                f"I have {len(indivs)} existing algorithms with their codes as follows:\n"\
                f"{prompt_indiv}"\
                "Please help me create a new algorithm that combines and builds upon the strengths of the given approaches.\n"\
                "Firstly, identify the common backbone idea in the provided algorithms. Then, create a new approach that inherits and enhances these patterns while introducing novel improvements. "\
                f"{self._base_prompt()}"
                
            return prompt_content
        elif self.mode == PromptMode.PROMPT:
            prompt_indiv = self._get_prompt_indivs(indivs)

            prompt_content = f"{self.task}\n"\
                f"I have {len(indivs)} existing prompt generation approaches with their functions as follows:\n"\
                f"{prompt_indiv}"\
                "Please help me create a new prompt generation approach that combines and builds upon the strengths of the given approaches.\n"\
                "First, identify the common successful patterns in the provided approaches. Then, create a new approach that inherits and enhances these patterns while introducing novel improvements. "\
                f"{self._base_prompt()}"
                        
            return prompt_content
        elif self.mode == PromptMode.TOOL:
            raise NotImplementedError

    def _get_prompt_m1(self, indiv: dict):
        prompt_indiv = self._get_prompt_indivs(indiv)
        if self.mode == PromptMode.CODE:
            prompt_content = f"{self.task}\n"\
                "I have one algorithm with its code as follows.\n"\
                f"{prompt_indiv}"\
                "Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided.\n"\
                f"{self._base_prompt()}"
            return prompt_content
        elif self.mode == PromptMode.PROMPT:
            prompt_content = f"{self.task}\n"\
                "I have one prompt generation approach with its function as follows.\n"\
                f"{prompt_indiv}"\
                "Please assist me in creating a new prompt generation approach that has a different form but can be a modified version of the approach provided.\n"\
                f"{self._base_prompt()}"
            return prompt_content
        elif self.mode == PromptMode.TOOL:
            raise NotImplementedError

    def _get_prompt_m2(self, indiv: dict):
        prompt_indiv = self._get_prompt_indivs(indiv)
        if self.mode == PromptMode.CODE:
            prompt_content = f"{self.task}\n"\
                "I have one algorithm with its code as follows.\n"\
                f"{prompt_indiv}"\
                "Please identify the main algorithm parameters and assist me in creating a new algorithm that has different parameter settings of the score function provided.\n"\
                f"{self._base_prompt()}"
            return prompt_content
        elif self.mode == PromptMode.PROMPT:
            prompt_content = f"{self.task}\n"\
                "I have one prompt generation approach with its function as follows.\n"\
                f"{prompt_indiv}"\
                "Please identify the main approach parameters and assist me in creating a new prompt generation approach that has different parameter settings of the prompt generation function provided.\n"\
                f"{self._base_prompt()}"
            return prompt_content
        elif self.mode == PromptMode.TOOL:
            raise NotImplementedError
        
    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "func_name": self.func_name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "input_types": self.input_types,
            "output_types": self.output_types,
            "mode": self.mode.value
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MetaPrompt':
        return cls(
            task=data["task"],
            func_name=data["func_name"] if "func_name" in data else data["name"],
            inputs=data["inputs"],
            outputs=data["outputs"],
            input_types=data["input_types"],
            output_types=data["output_types"],
            mode=get_prompt_mode(data["mode"])
        )
        
    @classmethod
    def from_json(cls, file_path: str) -> 'MetaPrompt':
        with open(file_path, 'r') as file:
            data = json.load(file)
        return cls.from_dict(data)


def clean_reasoning_str(reasoning: str):
    return reasoning.split("\n")[0].split("}")[0].strip()
        
def parse_evol_response(response: str):
    reasoning = re.findall(r"\{(.*)\}", response, re.DOTALL)
    if len(reasoning) == 0:
        if 'python' in response:
            reasoning = re.findall(r'^.*?(?=python)', response, re.DOTALL)
        elif 'import' in response:
            reasoning = re.findall(r'^.*?(?=import)', response, re.DOTALL)
        else:
            reasoning = re.findall(r'^.*?(?=def)', response, re.DOTALL)
            
    code = extract_python_funcions(response)
    
    
    return clean_reasoning_str(reasoning[0]) if reasoning else "", code if code else ""


# Plan as a Graph (Ideally, current version fall-back to a chain of plan ....)
pseudo_code_prompt = """ 
Generate a python function with sub-functions to complete the task. 
Can use pseudo code for sub-functions with explicit input, input types, output types, and comments, but no implementations.
""" 

PLAN_GRAPH_PROMPT = """Generate a JSON-style plan represented as a Directed Acyclic Graph (DAG) for the task. The plan should include:
- **Nodes**: Each node represents a key action or step and must contain the following attributes:
- `task`: Description of the task.
- `name`: Concise name used for the task function.
- `inputs`: List of input parameters needed to perform the action.
- `input_types`: List of corresponding types for each input parameter.
- `outputs`: List of output parameters produced by the action.
- `output_types`: List of corresponding types for each output parameter.
- `target`: The purpose or goal that the action contributes to.
- `mode`: The execution mode for this task ("CODE" or "PROMPT").

- **Edges**: Each edge represents a dependency or relationship between nodes, indicating that one step supports or leads to another.
- `source`: The `id` of the source node (the preceding action).
- `target`: The `id` of the target node (the subsequent action).

**Output Format:**

Provide the output in the following JSON structure:

```json
{
"nodes": [
    {
    "task": "Task 1",
    "name": "task_1"
    "inputs": inputs_str,
    "input_types": input_types_str,
    "outputs": ["output_11", "output_12"],
    "output_types": ["str", "str"],
    "target": "Purpose of Action 1"
    "mode": "CODE"
    },
    {
    "task": "Task 2",
    "name": "task_2",
    "inputs": ["input_21", "input_22"],
    "input_types": input_types_str,
    "outputs": outputs_str,
    "output_types": output_types_str,
    "target": "Purpose of Action 2",
    "mode": "PROMPT"
    }
    // Add more nodes as needed
],
"edges": [
    {
    "source": "task_1",
    "target": "task_2"
    }
    // Add more edges as needed
]
}
```
"""

PLAN_REQUIRED_KEYS = ["task", "name", "inputs", "input_types", "outputs", "output_types", "target", "mode"]
ALLOWED_EXTRA_KEYS = ['code', 'reasoning', 'fitness']

def _rectify_plan_dict(plan_dict: dict, meta_prompt: "MetaPlan"):
    main_func_name = meta_prompt.func_name
    err_msg = []
    
    seen_names = {}
    for node in plan_dict["nodes"]:
        if node.get("name", "") and node.get("name", "") != main_func_name:
            seen_names[node["name"]] = node
    nodes = list(seen_names.values())
    removed_nodes = set(node["name"] for node in plan_dict["nodes"]) - set(node["name"] for node in nodes)
    
    # 2. Remove edges connected to removed nodes
    edges = [edge for edge in plan_dict["edges"] 
             if edge["source"] not in removed_nodes and edge["target"] not in removed_nodes]
    
    # 3. Validate start and end nodes
    start_nodes = {node["name"] for node in nodes 
                  if not any(edge["target"] == node["name"] for edge in edges)}
    end_nodes = {node["name"] for node in nodes 
                if not any(edge["source"] == node["name"] for edge in edges)}
    
    # print("Start nodes: ", start_nodes)
    # print("End nodes: ", end_nodes)
    
    # Check input/output compatibility
    for node in nodes:
        if node["name"] in start_nodes:
            if node["input_types"] != meta_prompt.input_types: 
                err_msg.append(f"Start node {node['name']} has incompatible input types")
            elif node["inputs"] != meta_prompt.inputs:
                node["inputs"] = meta_prompt.inputs
        if node["name"] in end_nodes:
            if node["output_types"] != meta_prompt.output_types:
                err_msg.append(f"End node {node['name']} has incompatible output types")
            elif node["outputs"] != meta_prompt.outputs:
                node["outputs"] = meta_prompt.outputs
    
    # Update plan_dict with cleaned data
    plan_dict = {
        "nodes": nodes,
        "edges": edges
    }
    
    if len(err_msg) == 0:
        return plan_dict, ""
    else: # why would we ever return a mal-functioning plan_dict anyway ...
        return {}, err_msg
    
def check_n_rectify_plan_dict(plan_dict: dict, meta_prompt: "MetaPlan"):
    """
    1. Ensure all required keys are present in the plan_dict
    2. For edges which use ['task', 'name', 'id' (if exists)] as source or target, rectify them to using 'name' instead
    3. Clean up keys for 'nodes' to only contain required keys
    """
    err_msg = []
    
    if not isinstance(plan_dict, dict) or 'nodes' not in plan_dict or 'edges' not in plan_dict:
        err_msg.append("Invalid plan_dict structure: must be a dict with 'nodes' and 'edges' keys")
        return {}, ("\n").join(err_msg)
    
    # Check if all required keys are present in the nodes
    for node in plan_dict['nodes']:
        missing_keys = [key for key in PLAN_REQUIRED_KEYS if key not in node]
        if missing_keys:
            err_msg.append(f"Planning Node {node.get('name', 'unknown')} is missing required keys: {', '.join(missing_keys)}")
    if err_msg:
        return {}, ("\n").join(err_msg)
    
    # Rectify edges to use 'name' consistently
    id_to_name_map = {node.get('id', ''): node['name'] for node in plan_dict['nodes']}
    task_to_name_map = {node['task']: node['name'] for node in plan_dict['nodes']}
    name_set = {node['name'] for node in plan_dict["nodes"]}
    
    def rectify_edge_endpoint(endpoint: str):
        err_msg = ""
        if endpoint in id_to_name_map:
            return id_to_name_map[endpoint], err_msg
        elif endpoint in task_to_name_map:
            return task_to_name_map[endpoint], err_msg
        elif endpoint in name_set:
            return endpoint, err_msg
        else:
            err_msg = f"Planning EdgeEndpoint {endpoint} not found in plan_dict"
            return endpoint, err_msg
    
    for edge in plan_dict['edges']:
        source, err_msg_delta = rectify_edge_endpoint(edge['source'])
        if err_msg_delta:
            err_msg.append(err_msg_delta)
        target, err_msg_delta = rectify_edge_endpoint(edge['target'])
        if err_msg_delta:
            err_msg.append(err_msg_delta)
        edge["source"], edge["target"] = source, target
    
    if err_msg:
        return {}, ("\n").join(err_msg)
        
    # Clean up nodes, remove weird keys
    for node in plan_dict['nodes']:
        keys_to_remove = [key for key in node.keys() if key not in PLAN_REQUIRED_KEYS + ALLOWED_EXTRA_KEYS]
        for key in keys_to_remove:
            node.pop(key)
            
    plan_dict, err_msg_delta = _rectify_plan_dict(plan_dict, meta_prompt)
    if err_msg_delta:
        err_msg += err_msg_delta

    return plan_dict, ("\n").join(err_msg) if err_msg else ""
  

def get_plan_str(plan_dict: dict) -> str:
    # Build string representation of nodes
    nodes_str = "Nodes:\n"
    for node in plan_dict['nodes']:
        nodes_str += f"- Task: {node['task']}\n"
        nodes_str += f"  Name: {node['name']}\n"
        nodes_str += f"  Inputs: {', '.join(node['inputs'])} ({', '.join(node['input_types'])})\n"
        nodes_str += f"  Outputs: {', '.join(node['outputs'])} ({', '.join(node['output_types'])})\n\n"
    
    # Build string representation of execution flow
    edges_str = "Execution Flow:\n"
    for edge in plan_dict['edges']:
        edges_str += f"- {edge['source']} → {edge['target']}\n"
    
    return nodes_str + edges_str


def generate_test_cases_template(plan_dict: dict, main_test_cases: list) -> str:
    test_cases_list = []
    inputs = [io[0] for io in main_test_cases]
    outputs = [io[1] for io in main_test_cases]
    
    for i, node in enumerate(plan_dict['nodes']):
        node_test_case = {
            "name": node['name'],
            "inputs": [],
            "outputs": []
        }
        
        # Handle entry node (first node)
        if i == 0:
            node_test_case["inputs"] = inputs
            node_test_case["outputs"] = [{"result": "..."} for _ in inputs]
        # Handle exit node (last node)
        elif i == len(plan_dict['nodes']) - 1:
            node_test_case["inputs"] = [{"google_result": "..."} for _ in inputs]
            node_test_case["outputs"] = outputs
        # Handle intermediate nodes
        else:
            node_test_case["inputs"] = [{input_name: "..." for input_name in node['inputs']} for _ in inputs]
            node_test_case["outputs"] = [{output_name: "..." for output_name in node['outputs']} for _ in inputs]
            
        test_cases_list.append(node_test_case)
    
    return f"```json\n{json.dumps(test_cases_list, indent=4)}\n```"


def _filter_test_cases_list(test_cases_list: list, main_test_cases: list, accept_extra: bool = False):
    """Filter and validate test cases against main test cases.
    
    Args:
        test_cases_list: List of dictionaries containing test cases for each step
        main_test_cases: List of tuples containing (input, output) pairs to validate against
        accept_extra: Whether to keep additional test cases beyond the main ones
    
    Returns:
        Dictionary mapping step names to filtered test cases
    """
    err_msg = []
    
    # Find indices of main test cases in the first step
    match_indices = []
    match_case_indices = []
    first_step = test_cases_list[0]
    last_step = test_cases_list[-1]
    for main_case in main_test_cases:
        main_input = main_case[0]
        for i, test_input in enumerate(first_step['inputs']):
            if test_input == main_input:
                test_output = last_step['outputs'][i]
                main_output = main_case[1]
                if test_output == main_output:
                    match_indices.append(i)
                    match_case_indices.append(main_case)
                else:
                    err_msg.append(f"Output mismatch for input {main_input}, expected {main_output}, got {test_output}")
                break
        
    filtered_list = []
    for step in test_cases_list:
        filtered_step = {
            'name': step['name'],
            'inputs': [step['inputs'][i] for i in match_indices],
            'outputs': [step['outputs'][i] for i in match_indices]
        }
        filtered_list.append(filtered_step)
        
    if len(filtered_list) == 0:
        err_msg.append("No matching test cases with main test cases found")
                    
    return filtered_list, "\n".join(err_msg) if err_msg else ""


def combine_test_cases_list(t1, t2, unique=True):
    # combining test cases list from different runs ...
    combined_t = []
    for (sub_node_cases1, sub_node_cases2) in zip(t1, t2):
        combined_cases = {"name": sub_node_cases1["name"],
                        "inputs": sub_node_cases1["inputs"] + sub_node_cases2["inputs"],
                        "outputs": sub_node_cases1["outputs"] + sub_node_cases2["outputs"]}
        combined_t.append(combined_cases)
        
    # filter unique inputs ...
    if unique:
        unique_indices = [] 
        unique_initial_inputs = []
        first_step_inputs = combined_t[0]['inputs']
        for i, input_dict in enumerate(first_step_inputs):
            if input_dict not in unique_initial_inputs:
                unique_indices.append(i)
                unique_initial_inputs.append(input_dict)
            else:
                continue
            
        for sub_node_cases in combined_t:
            sub_node_cases["inputs"] = [sub_node_cases["inputs"][i] for i in unique_indices]
            sub_node_cases["outputs"] = [sub_node_cases["outputs"][i] for i in unique_indices]
        
    return combined_t


def _spawn_test_cases(plan_dict: dict, main_test_cases: list, get_response: Callable) -> tuple[list, str]:
    # generating prompt for spawning sub-node test cases
    plan_str = get_plan_str(plan_dict)

    oneshot_prompt = generate_test_cases_template(plan_dict, main_test_cases)

    spawn_prompt = f"Here is a execution plan for a function: \n{plan_str}\n\n help generate test cases for each sub-function by filling the ... with proper inputs and outputs, output JSON like this: \n{oneshot_prompt}"
    response = get_response(spawn_prompt)

    try: 
        test_cases_list = extract_json_from_text(response)
        filtered_list, err_msg = _filter_test_cases_list(test_cases_list, main_test_cases)
        return filtered_list, err_msg
    except Exception as e:
        err_msg = f"Error in spawning test cases: {e}"
        return [], err_msg
    
def _build_test_cases_dict(spawned_test_cases: list):
    output_dict = {}
    for cases in spawned_test_cases:
        sub_node_test_cases = [(i, o) for i, o in zip(cases["inputs"], cases["outputs"])]
        output_dict[cases["name"]] = sub_node_test_cases
    return output_dict 


def spawn_test_cases_sequential(plan_dict: dict, main_test_cases: list, get_response: Callable, max_tries: int = 6) -> tuple[dict, str]: # Deprecated
    test_case_count = len(main_test_cases)
    test_cases_list = []
    err_msg = []
    for _ in range(max_tries):
        test_cases_list_delta, err_msg_delta = _spawn_test_cases(plan_dict, main_test_cases, get_response)
        if err_msg_delta == "" and test_cases_list != []:
            test_cases_list = combine_test_cases_list(test_cases_list, test_cases_list_delta)
        elif err_msg_delta == "" and test_cases_list == []:
            test_cases_list = test_cases_list_delta
        else:
            err_msg.append(err_msg_delta)
        
        if test_cases_list and len(test_cases_list[0]["inputs"]) >= test_case_count:
            break

    if test_cases_list:
        return _build_test_cases_dict(test_cases_list), "\n".join(err_msg) if err_msg else ""
    else:
        return {}, "\n".join(err_msg) if err_msg else ""
    
    
def spawn_test_cases(plan_dict: dict, main_test_cases: list, get_response: Callable, batch_size: int = 1, unique=True) -> tuple[dict, str]:

    plan_str = get_plan_str(plan_dict)
    oneshot_prompt = generate_test_cases_template(plan_dict, main_test_cases)
    spawn_prompt = f"Here is a execution plan for a function: \n{plan_str}\n\n help generate test cases for each sub-function by filling the ... with proper inputs and outputs, output JSON like this: \n{oneshot_prompt} \nUSE THIS JSON AS A base. ONLY change THE ... and do not change any other values that are set, eventhought they are not diverse or people will die from your mistakes. Fill up the ... based on the already set values, which are the true input and output of the entire of the entire plan. You can imagine it as if you are filling ... based on the intermediate values of the given true inputs/outputs so do not create your own input/output based on anything else or people will die from you"
    spawn_prompts = [spawn_prompt] * batch_size
    responses = get_response(spawn_prompts)
    test_cases_deltas = []
    err_msgs = []
    test_cases_list = None
    for response in responses:
        try: 
            test_cases_list = extract_json_from_text(response)
            filtered_list, err_msg = _filter_test_cases_list(test_cases_list, main_test_cases) # add check-up for incomplete test_cases
            if filtered_list:
                test_cases_deltas.append(filtered_list)
            err_msgs.append(err_msg)
        except Exception as e:
            err_msgs.append(f"Error in spawning test cases: {e}")
        
    for i, test_cases_delta in enumerate(test_cases_deltas):
        if i == 0:
            test_cases_list = test_cases_delta
        else:
            test_cases_list = combine_test_cases_list(test_cases_list, test_cases_delta, unique)
    if test_cases_list:
        return _build_test_cases_dict(test_cases_list), "\n".join(err_msgs) if err_msgs else ""
    else:
        return {}, "\n".join(err_msgs) if err_msgs else ""
    

# For evaluation node, external memory is important (it could access local files through its code-interpreter, a skill which it should learn in the process)
# How do we improve on the evaluation? Can we evaluate on the evaluation result? 

eval_goal_prompt = """
Generate a JSON response describing a task to evaluate whehter the goal is completed.

**Output Format:**

Provide the output in the following JSON structure:
```json
{
    "task": "Evaluate Task",
    "name": "eval_goal",
    "input": "Inputs required for evaluation",
    "output": "Outputs required for evaluation",
    "target": "Purpose of the evaluation",
    "mode: "CODE"
}
"""


# MetaPlan decompose a task into chained sub-tasks (output-intput chained)
@dataclass
class MetaPlan:
    task: str
    func_name: str
    inputs: list
    outputs: list
    input_types: list 
    output_types: list
    
    @property 
    def _base_pseudo_code_prompt(self):
        prompt_content = f"First, describe your new algorithm and main steps in one sentence."\
            f"The description must be inside a brace. Next implement it in Python as a pseudo function named {self.func_name}."\
            f"This function should accept {len(self.inputs)} input(s): {', '.join(self.inputs)} with types {', '.join(self.input_types)}. "\
            f"The function should return {len(self.outputs)} output(s): {', '.join(self.outputs)} with types {', '.join(self.output_types)}. "\
            "Include sub-functions with explicit input, input types, output types, and comments."\
            "Can use pseudo code for sub-functions with explicit input, input types, output types, and comments, but no implementations."\
            "Example: \n```python\n#Your implementation\n```\n"
        return prompt_content
    
    @property
    def _base_plan_graph_prompt(self):
        input_str = f"[{', '.join(self.inputs)}]"
        input_types_str = f"[{', '.join(self.input_types)}]"
        output_str = f"[{', '.join(self.outputs)}]"
        output_types_str = f"[{', '.join(self.output_types)}]"
        prompt_content = PLAN_GRAPH_PROMPT.replace("inputs_str", input_str).replace("input_types_str", input_types_str).replace("outputs_str", output_str).replace("output_types_str", output_types_str)
        return prompt_content
    
    def _get_prompt_i1(self, feedback: str = "", parents: list = []):
        prompt_content = f"Task: {self.task}\n{self._base_pseudo_code_prompt}"
        if feedback:
            prompt_content += f"\nPlease incorporate this feedback in your solution: {feedback}"
        return prompt_content
    
    def _get_prompt_m1(self, feedback: str = "", parents: list = []):
        raise NotImplementedError("Method m1 not implemented")
        
    def _get_prompt_m2(self, feedback: str = "", parents: list = []):
        raise NotImplementedError("Method m2 not implemented")
    
    def _get_prompt_e1(self, feedback: str = "", parents: list = []):
        raise NotImplementedError("Method e1 not implemented")
    
    def _get_prompt_e2(self, feedback: str = "", parents: list = []):
        raise NotImplementedError("Method e2 not implemented")
    
    def _get_plan_graph_prompt(self, code: str):
        prompt_content = f"Task: {self.task}\n\n"
        prompt_content += f"Pseudo Code:\n{code}\n\n"
        prompt_content += self._base_plan_graph_prompt
        return prompt_content
    
    def to_dict(self):
        return {
            "task": self.task,
            "func_name": self.func_name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "input_types": self.input_types,
            "output_types": self.output_types
        }
        
    @classmethod 
    def from_dict(cls, data: dict):
        return cls(
            task=data["task"],
            func_name=data["func_name"],
            inputs=data["inputs"],
            outputs=data["outputs"],
            input_types=data["input_types"],
            output_types=data["output_types"]
        )    


def build_graph_from_json(parsed_json):
    """ 
    Construct graph object from parsed json
    - nodes & edges as the key values
    """

    G = nx.DiGraph()

    # Add nodes
    for node in parsed_json['nodes']:
        try:
            G.add_node(node)
        except:
            G.add_node(node['id'], label=node['label'])
            
    # Add edges
    for edge in parsed_json['edges']:
        G.add_edge(edge['from'], edge['to'], label=edge['relationship'])

    return G 

def extract_json_from_text(text):
    """
    Extracts a JSON object from a text containing either a JSON code block or a JSON-like structure.
    
    Parameters:
        text (str): The input text containing the JSON code block or JSON-like structure.
        
    Returns:
        dict: The parsed JSON object.
        
    Raises:
        ValueError: If no JSON structure is found or JSON is invalid.
    """
    # Available Patterns
    code_json_pattern = r'```json\s*(\{.*?\})\s*```'
    code_python_pattern = r'```python\s*(.*?)\s*```'
    json_list_pattern = r'```json\s*(.*?)\s*```'
    json_dict_pattern = r'\{[^}]+\}'
    
    code_json_match = re.search(code_json_pattern, text, re.DOTALL)
    code_python_match = re.search(code_python_pattern, text, re.DOTALL)
    list_match = re.search(json_list_pattern, text, re.DOTALL)
    dict_match = re.search(json_dict_pattern, text, re.DOTALL)
    
    if code_json_match:
        json_str = code_json_match.group(1)
    elif code_python_match:
        json_str = code_python_match.group(1)
    elif list_match:
        json_str = list_match.group(1)
    elif dict_match:
        json_str = dict_match.group(0)
    else:
        raise ValueError("No JSON structure found in the provided text.")
          
    # return json_str
    # json_str = json_str.replace("'", '"')
    error_msg = ""
    try:
        json_data = json.loads(json_str)
        return json_data 
    except json.JSONDecodeError as e:
        error_msg += f"JsonDecodeError : \n{e}"
    try:
        json_data = ast.literal_eval(json_str)
        return json_data
    except Exception as e:
        error_msg += f"AstLiteralError : \n{e}"
        
    raise ValueError(error_msg)


def extract_python_code(response):
    code_python_pattern = r'```python\s*(.*?)\s*```'
    code_match = re.search(code_python_pattern, response, re.DOTALL)
    if code_match:
        code = code_match.group(1)
        return code 
    else:
        raise ValueError("No code block found in the response.")
    
    
def extract_imports_and_functions(code_str):
    tree = ast.parse(code_str)
    imports = []
    functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            names = ", ".join(alias.name for alias in node.names)
            imports.append(f"from {node.module} import {names}")
        elif isinstance(node, ast.FunctionDef):
            if not hasattr(node, "parent"): functions.append(ast.unparse(node))
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    child.parent = node
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    child.parent = node
    
    return imports, functions


def extract_python_funcions(response: str) -> str:
    """ 
    Extract python snippet, use ast to extract functions and imports
    """
    code_str = extract_python_code(response)
    imports, functions = extract_imports_and_functions(code_str)
    code = "\n".join(imports + functions)
    return code


def decide_node_class(node: dict) -> str:
    if "code" in node and node["code"] != "": 
        return "retrieved node"
    else:
        return "hypothetical node"


def parse_plan_graph(plan_dict: dict) -> dict:
    # Create a DAG structure compatible with the existing visualization function
    dag = {}
    for node in plan_dict["nodes"]:
        node_id = node["name"]
        task_str = f"Task: {node['task']}\nInput: {node['inputs']}\nOutput: {node['outputs']}\nTarget: {node['target']}\nMode: {node['mode']}"
        node_class = decide_node_class(node)
        dag[node_id] = {
            "name": node["name"],
            "type": "code" if node["mode"] == "CODE" else "llm",
            "opacity": 1.0,
            "importance": 1.0,
            "edges": [],
            "task_str": task_str,
            "code_str": "",  # Add the code_str field, initially empty
            "class": node_class,
            "task": node["task"]
        }
    
    # Add edges to the DAG
    for edge in plan_dict["edges"]:
        source = edge["source"]
        target = edge["target"]
        if source in dag and target in dag:
            dag[source]["edges"].append(target)
    
    return dag


ALIGNMENT_CHECK_PROMPT = """
Compare the following two dictionary outputs and determine if they are essentially aligned:

Predicted output: {pred_output}
Target output: {target_output}

Consider them aligned if:
1. They have the same keys.
2. The values for each key are 'basically the same':
   - For numbers: exactly the same
   - For text: convey the same meaning, even if worded differently
   - For booleans: exactly the same
   - For lists/dicts: contents are similar

Respond in this format:
{{
  "aligned": true/false,
  "comment": "Whatever you want to say on the prediction, be concise."
}}
"""

GENERATE_NODES_FROM_API = """Generate a JSON-style list representing new nodes based on the API documentation, which can be resuable for more nodes. These nodes have to be general but are not one liners (make them slightly complex/useful). You can also make nodes which are just examples in the documentation (recommended to have some as they are most likely error free). The list should include:
- **Nodes**: Each node represents a key action or step and must contain the following attributes:
- `task`: Description of the task. MAKE SURE THE TASK IS NOT JUST A ONE LINER BUT ALSO VERY USEFUL
- `name`: Concise name used for the task function.
- `inputs`: List of input parameters needed to perform the action.
- `input_types`: List of corresponding types for each input parameter.
- `outputs`: List of output parameters produced by the action.
- `output_types`: List of corresponding types for each output parameter.
- `target`: The purpose or goal that the action contributes to.
- `mode`: The execution mode for this task ("CODE" or "PROMPT").
- 'relevant_docs': Relevant documentation for the task based on the given documentation. Make sure this is very verbose, stating what library you are using, suggest the functions and give its signatures and meaning or anything that a developer would need to code out/use the library without checking the internet. For example, give example function calls which could be useful as well as how to import the function
**Output Format:**

To be direct, have 2 nodes which are copy pasted from the documentation if possible which can be use widely. Have more nodes which are more complex and can be build on the first 2 nodes

Provide the output in the following JSON structure:

```json
{
"nodes": [
    {
    "task": "Task 1",
    "name": "task_1"
    "inputs": inputs_str,
    "input_types": input_types_str,
    "outputs": ["output_11", "output_12"],
    "output_types": ["str", "str"],
    "target": "Purpose of Action 1"
    "mode": "CODE",
    "relevant_docs": "The function xyz takes in the following parameters and returns the following output."
    },
    {
    "task": "Task 2",
    "name": "task_2",
    "inputs": ["input_21", "input_22"],
    "input_types": input_types_str,
    "outputs": outputs_str,
    "output_types": output_types_str,
    "target": "Purpose of Action 2",
    "mode": "PROMPT",
    "relevant_docs": "The function abc does this and that."
    }
    // Add more nodes as needed
]
}
```\n
If no json is given people will die. Make sure the nodes are somewhat complex/ not one-liners of calling the api function\n
"""

CHOOSE_USEFUL_LINKS = """You are a master programmer. You have to help a junior programmer go through some doucmentation, but due to time constraits, he should not look at useless links.
Given a list of links, choose the most useful links which are most likely to be useful for the junior programmer to code without ever searching the documentation himself. 
Provide the output in the following JSON structure. DO NOT GIVE THE CODE FOR IT but the JSON:
```json
{
    "links": [0, 1, 2, 3] //indexes of chosen linksxs
}
```
Anything without code such as forum or community pages or faq or installation pages are useless. Remove them based on the link names
"""