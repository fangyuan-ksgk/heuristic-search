from dataclasses import dataclass
from enum import Enum
import re
import json
import re
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
    input: str
    output: str
    input_types: list 
    output_types: list
    mode: PromptMode
    
    @property
    def joined_inputs(self):
        if len(self.input) > 1:
            return ", ".join("'" + s + "'" for s in self.input)
        else:
            return "'" + self.input[0] + "'"
        
    @property
    def joined_outputs(self):
        if len(self.output) > 1:
            return ", ".join("'" + s + "'" for s in self.output)
        else:
            return "'" + self.output[0] + "'"
        
    def _base_prompt(self):
        if self.mode == PromptMode.CODE:
            prompt_content = f"First, describe your new algorithm and main steps in one sentence. "\
                "The description must be inside a brace. Next, implement it in Python as a function named "\
                f"{self.func_name}. This function should accept {len(self.input)} input(s): "\
                f"{self.joined_inputs} with types {', '.join(self.input_types)}. "\
                f"The function should return {len(self.output)} output(s): "\
                f"{self.joined_outputs} with types {', '.join(self.output_types)}."\
                "Make sure to include type hints in your function signature."
            return prompt_content
        elif self.mode == PromptMode.PROMPT:
            prompt_content = f"First, describe your new reasoning and main thoughts in one sentence."\
                "The description must be inside a brace. Implement a Python function that generates a prompt to guide an AI in completing the task. "\
                f"Follow these specifications: - Function name: generate_prompt - Input parameters: {self.joined_inputs} - Return value: A string containing the final prompt for the AI."\
                "Ask for JSON-style response with Output values: {self.joined_outputs}"\
                "Your function should incorporate the reasoning from step 1 and use the input parameters to create a tailored prompt for the task."\
                "Specify types for input and output."
            return prompt_content
        elif self.mode == PromptMode.TOOL:
            raise NotImplementedError
        
    def _get_eval_prompt(self):
        """ 
        Asking for (input, output) pairs for evaluation
        """
        prompt_content = f"Task: {self.task}\n\n"\
                         f"For the Python function '{self.func_name}', generate 5 diverse (input, output) pairs for evaluation. "\
                         "These pairs should cover different scenarios, including edge cases.\n\n"\
                         "Function signature:\n"\
                         f"def {self.func_name}({', '.join(f'{inp}: {type_hint}' for inp, type_hint in zip(self.input, self.input_types))}) -> {self.output_type}:\n\n"\
                         "Provide your response as a Python list of dictionaries. Each dictionary should have 'input' and 'expected_output' keys. "\
                         "Ensure that the types match the function signature.\n\n"\
                         "Example format:\n"\
                         "[\n"\
                         "    {\n"\
                         "        'input': {" + ', '.join(f"'{inp}': {type_hint}(...)" for inp, type_hint in zip(self.input, self.input_types)) + "},\n"\
                         "        'expected_output': {self.output_type}(...)\n"\
                         "    },\n"\
                         "    ...\n"\
                         "]\n\n"\
                         "Provide 5 such pairs, ensuring type correctness and diversity in the inputs and outputs."        
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
                f"{self.func_name}. This function should accept {len(self.input)} input(s): "\
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
            "input": self.input,
            "output": self.output,
            "mode": self.mode.value
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MetaPrompt':
        return cls(
            task=data["task"],
            func_name=data["func_name"],
            input=data["input"],
            output=data["output"],
            mode=PromptMode(data["mode"])
        )

        
        
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
    
    return reasoning[0].strip() if reasoning else "", code[0].strip() if code else ""


# Plan as a Graph (Ideally, current version fall-back to a chain of plan ....)
plan_graph_prompt = """
Generate a JSON-style plan represented as a Directed Acyclic Graph (DAG) to achieve the goal. Use creative topology in the DAG, include parallel tasks if required.

The plan should include:

- **Nodes**: Each node represents a key action or step and must contain the following attributes:
  - `task`: Description of the task.
  - `name`: Concise name used for the task function.
  - `input`: The resources, information, or prerequisites needed to perform the action.
  - `output`: The immediate result or outcome of the action.
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
      "input": "Inputs required for Action 1",
      "output": "Outputs/result of Action 1",
      "target": "Purpose of Action 1"
      "mode": "CODE"
    },
    {
      "task": "Task 2",
      "name": "task_2",
      "input": "Inputs required for Action 2",
      "output": "Outputs/result of Action 2",
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
    
    def _get_prompt_i1(self):
        prompt_content = f"Goal: {self.goal}\n{self._base_prompt}"
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
    Extracts a JSON object from a text containing a JSON code block.
    
    Parameters:
        text (str): The input text containing the JSON code block.
        
    Returns:
        dict: The parsed JSON object.
        
    Raises:
        ValueError: If no JSON code block is found or JSON is invalid.
    """
    # Regular expression to find JSON code block enclosed in ```json ... ```
    json_block_pattern = r'```json\s*(\{.*?\})\s*```'
    
    # Use re.DOTALL to allow '.' to match newline characters
    match = re.search(json_block_pattern, text, re.DOTALL)
    
    if not match:
        raise ValueError("No JSON code block found in the provided text.")
    
    json_str = match.group(1)
    
    try:
        # Parse the JSON string into a Python dictionary
        json_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON content: {e}")
    
    return json_data


