from dataclasses import dataclass
from enum import Enum
import re
import json
import re
import ast
import networkx as nx
from dataclasses import dataclass
from typing import Optional, Union

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
                f"Follow these specifications: - Function name: generate_prompt - Input parameters: {self.joined_inputs} - Return value: A string containing the final prompt for the AI. "
                f"Ask for JSON-style response with output dictionary: {{" + ', '.join(f"'{out}': {type_hint}(...)" for out, type_hint in zip(self.outputs, self.output_types)) + "}}\n"
                "Your function should incorporate the reasoning from step 1 and use the input parameters to create a tailored prompt for the task. ")
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
    
    def _get_pseudo_code_prompt(self, feedback: str = ""):
        prompt_content = f"Task: {self.task}\n{self._base_pseudo_code_prompt}"
        if feedback:
            prompt_content += f"\nPlease incorporate this feedback in your solution: {feedback}"
        return prompt_content
    
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
    
    # e1/e2/m1/m2 to be implemented


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
        print("No code block found in the response.")
        return ""
    
    
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
            functions.append(ast.unparse(node))
    
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
