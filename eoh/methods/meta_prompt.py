from dataclasses import dataclass
from enum import Enum
import re
import json
import re
import ast
import networkx as nx
from dataclasses import dataclass

class PromptMode(Enum):
    CODE = "code"
    TOOL = "tool"
    PROMPT = "prompt"
    
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
        
    def _get_prompt_i1(self):
        prompt_content = f"{self.task}\n{self._base_prompt()}"
        return prompt_content
        
    def _get_prompt_e1(self, indivs: list):
        if self.mode == PromptMode.CODE:
            prompt_indiv = ""
            for i, indiv in enumerate(indivs, 1):
                prompt_indiv += f"No.{i} algorithm and the corresponding code are:\n{indiv['algorithm']}\n{indiv['code']}\n"

            prompt_content = f"{self.task}\n"\
                f"I have {len(indivs)} existing algorithms with their codes as follows:\n"\
                f"{prompt_indiv}"\
                "Please help me create a new algorithm that has a totally different form from the given ones.\n"\
                f"{self._base_prompt()}"
            return prompt_content
        elif self.mode == PromptMode.PROMPT:
            prompt_indiv = ""
            for i, indiv in enumerate(indivs, 1):
                prompt_indiv += f"No.{i} reasoning and the corresponding prompt generation function are:\n{indiv['reasoning']}\n{indiv['prompt_function']}\n"

            prompt_content = f"{self.task}\n"\
                f"I have {len(indivs)} existing prompt generation approaches with their functions as follows:\n"\
                f"{prompt_indiv}"\
                "Please help me create a new prompt generation approach that is totally different from the given ones.\n"\
                f"{self._base_prompt()}"
            return prompt_content
        
    def _get_prompt_e2(self, indivs: list):
        if self.mode == PromptMode.CODE:
            prompt_indiv = ""
            for i, indiv in enumerate(indivs, 1):
                prompt_indiv += f"No.{i} algorithm and the corresponding code are:\n{indiv['algorithm']}\n{indiv['code']}\n"

            prompt_content = f"{self.task}\n"\
                f"I have {len(indivs)} existing algorithms with their codes as follows:\n"\
                f"{prompt_indiv}"\
                "Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them.\n"\
                "Firstly, identify the common backbone idea in the provided algorithms. Secondly, based on the backbone idea describe your new algorithm in one sentence. "\
                "The description must be inside a brace. Thirdly, implement it in Python as a function named "\
                f"{self.func_name}. This function should accept {len(self.inputs)} input(s): "\
                f"{self.joined_inputs}. The function should return {len(self.output)} output(s): "\
                f"{self.joined_outputs}. {self.inout_inf} "\
                f"{self.other_inf}\n"\
                "Do not give additional explanations."
            return prompt_content
        elif self.mode == PromptMode.PROMPT:
            prompt_indiv = ""
            for i, indiv in enumerate(indivs, 1):
                prompt_indiv += f"No.{i} reasoning and the corresponding prompt generation function are:\n{indiv['reasoning']}\n{indiv['prompt_function']}\n"

            prompt_content = f"{self.task}\n"\
                f"I have {len(indivs)} existing prompt generation approaches with their functions as follows:\n"\
                f"{prompt_indiv}"\
                "Please help me create a new prompt generation approach that has a totally different form from the given ones but can be motivated from them.\n"\
                "Firstly, identify the common backbone idea in the provided approaches. Secondly, based on the backbone idea describe your new approach in one sentence. "\
                "The description must be inside a brace. Thirdly, implement a Python function that generates a prompt to guide an AI in completing the task. "\
                f"Follow these specifications: - Function name: generate_prompt - Input parameters: {self.joined_inputs} - Return value: A string containing the final prompt for the AI.\n"\
                "Your function should incorporate the reasoning from step 2 and use the input parameters to create a tailored prompt for the task."
            return prompt_content
        elif self.mode == PromptMode.TOOL:
            raise NotImplementedError

    def _get_prompt_m1(self, indiv: dict):
        if self.mode == PromptMode.CODE:
            prompt_content = f"{self.task}\n"\
                "I have one algorithm with its code as follows.\n"\
                f"Algorithm description: {indiv['algorithm']}\n"\
                f"Code:\n{indiv['code']}\n"\
                "Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided.\n"\
                f"{self._base_prompt()}"
            return prompt_content
        elif self.mode == PromptMode.PROMPT:
            prompt_content = f"{self.task}\n"\
                "I have one prompt generation approach with its function as follows.\n"\
                f"Reasoning: {indiv['reasoning']}\n"\
                f"Function:\n{indiv['prompt_function']}\n"\
                "Please assist me in creating a new prompt generation approach that has a different form but can be a modified version of the approach provided.\n"\
                f"{self._base_prompt()}"
            return prompt_content
        elif self.mode == PromptMode.TOOL:
            raise NotImplementedError

    def _get_prompt_m2(self, indiv: dict):
        if self.mode == PromptMode.CODE:
            prompt_content = f"{self.task}\n"\
                "I have one algorithm with its code as follows.\n"\
                f"Algorithm description: {indiv['algorithm']}\n"\
                f"Code:\n{indiv['code']}\n"\
                "Please identify the main algorithm parameters and assist me in creating a new algorithm that has different parameter settings of the score function provided.\n"\
                f"{self._base_prompt()}"
            return prompt_content
        elif self.mode == PromptMode.PROMPT:
            prompt_content = f"{self.task}\n"\
                "I have one prompt generation approach with its function as follows.\n"\
                f"Reasoning: {indiv['reasoning']}\n"\
                f"Function:\n{indiv['prompt_function']}\n"\
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
            func_name=data["func_name"],
            inputs=data["inputs"],
            outputs=data["outputs"],
            input_types=data["input_types"],
            output_types=data["output_types"],
            mode=PromptMode(data["mode"])
        )
        


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
            
    # Updated code extraction
    code = re.findall(r"((?:import|def).*?(?:^return.*?$|^    return.*?$))", response, re.DOTALL | re.MULTILINE)
    
    return clean_reasoning_str(reasoning[0]) if reasoning else "", code[0].strip() if code else ""


# Plan as a Graph (Ideally, current version fall-back to a chain of plan ....)
pseudo_code_prompt = """ 
Generate a python function with sub-functions to complete the goal. 
Can use pseudo code for sub-functions with explicit input, input types, output types, and comments, but no implementations.
"""


plan_graph_prompt = """
Generate a JSON-style plan represented as a Directed Acyclic Graph (DAG) to achieve the goal. Use creative topology in the DAG, include parallel tasks if required.

The plan should include:

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
      "inputs": ["input_11", "input_12", "input_13"],
      "input_types": ["str", "str", "str"],
      "outputs": ["output_11", "output_12", "output_13"],
      "output_types": ["str", "str", "str"],
      "target": "Purpose of Action 1"
      "mode": "CODE"
    },
    {
      "task": "Task 2",
      "name": "task_2",
      "inputs": ["input_21", "input_22", "input_23"],
      "input_types": ["str", "str", "str"],
      "outputs": ["output_21", "output_22", "output_23"],
      "output_types": ["str", "str", "str"],
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

@dataclass
class MetaPlan:
    goal: str
    
    @property 
    def _pseudo_code_prompt(self):
        prompt_content = pseudo_code_prompt
        return prompt_content
    
    @property
    def _base_prompt(self):
        prompt_content = f"First, describe the intuition for your tactics and main steps in one sentence. "\
                "The description must be inside a brace."\
                f"{plan_graph_prompt}"
        return prompt_content
    
    @property
    def _eval_prompt(self):
        prompt_content = f"First, describe the intuition for your tactics and main steps in one sentence. "\
                "The description must be inside a brace."\
                f"{eval_goal_prompt}"
        return prompt_content
    
    def _get_prompt_i1_pseudo_code(self):
        prompt_content = f"Goal: {self.goal}\n{self._pseudo_code_prompt}"
        return prompt_content
    
    def _get_prompt_i1(self):
        prompt_content = f"Goal: {self.goal}\n{self._base_prompt}"
        return prompt_content
    
    def _get_prompt_i1_with_pseudo_code(self, code: str):
        prompt_content = f"Goal: {self.goal}\n\n"
        prompt_content += f"Pseudo Code:\n{code}\n\n"
        prompt_content += self._base_prompt
        return prompt_content
    
    def _get_eval_prompt_i1(self):
        prompt_content = f"Goal: Evaluating wether the goal {self.goal} has been achieved.\n{self._eval_prompt}"
        return prompt_content
    
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


def parse_plan_graph(plan_dict: dict) -> dict:
    # Create a DAG structure compatible with the existing visualization function
    dag = {}
    for node in plan_dict["nodes"]:
        node_id = node["name"]
        task_str = f"Task: {node['task']}\nInput: {node['inputs']}\nOutput: {node['outputs']}\nTarget: {node['target']}\nMode: {node['mode']}"
        dag[node_id] = {
            "name": node["name"],
            "type": "code" if node["mode"] == "CODE" else "llm",
            "opacity": 1.0,
            "importance": 1.0,
            "edges": [],
            "task_str": task_str,
            "code_str": ""  # Add the code_str field, initially empty
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
