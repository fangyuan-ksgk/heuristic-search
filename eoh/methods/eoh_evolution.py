from abc import ABC, abstractmethod

#################################
#   Evolution on Graph          #
#   - Node: Code, Prompt, Tool  #
#   - Edge: Not Decided Yet     #
#################################
    

class PromptMode(Enum):
    CODE = "code"
    TOOL = "tool"
    PROMPT = "prompt"
    
# MetaPrompt describe meta-heuristic for each node
@dataclass
class MetaPrompt:
    task: str
    func_name: str
    input: str
    output: str
    
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
        
    def _base_prompt(self, mode: PromptMode):
        if mode == PromptMode.CODE:
            prompt_content = f"First, describe your new algorithm and main steps in one sentence. "\
                "The description must be inside a brace. Next, implement it in Python as a function named "\
                f"{self.func_name}. This function should accept {len(self.input)} input(s): "\
                f"{self.joined_inputs}. The function should return {len(self.output)} output(s): "\
                f"{self.joined_outputs}. {self.inout_inf} "\
                f"{self.other_inf}\n"\
                "Do not give additional explanations."
            return prompt_content
        elif mode == PromptMode.PROMPT:
            prompt_content = f"First, describe your new reasoning and main thoughts in one sentence."\
                "The description must be inside a brace. Implement a Python function that generates a prompt to guide an AI in completing the task. "\
                f"Follow these specifications: - Function name: generate_prompt - Input parameters: {self.joined_inputs} - Return value: A string containing the final prompt for the AI.\n" \
                "Your function should incorporate the reasoning from step 1 and use the input parameters to create a tailored prompt for the task."
            return prompt_content
        elif mode == PromptMode.TOOL:
            raise NotImplementedError
        
    def _get_prompt_i1(self, mode: PromptMode):
        prompt_content = f"{self.task}\n{self._base_prompt(mode)}"
        return prompt_content
        
    def _get_prompt_e1(self, indivs: list, mode: PromptMode):
        if mode == PromptMode.CODE:
            prompt_indiv = ""
            for i, indiv in enumerate(indivs, 1):
                prompt_indiv += f"No.{i} algorithm and the corresponding code are:\n{indiv['algorithm']}\n{indiv['code']}\n"

            prompt_content = f"{self.task}\n"\
                f"I have {len(indivs)} existing algorithms with their codes as follows:\n"\
                f"{prompt_indiv}"\
                "Please help me create a new algorithm that has a totally different form from the given ones.\n"\
                f"{self._base_prompt(mode)}"
            return prompt_content
        elif mode == PromptMode.PROMPT:
            prompt_indiv = ""
            for i, indiv in enumerate(indivs, 1):
                prompt_indiv += f"No.{i} reasoning and the corresponding prompt generation function are:\n{indiv['reasoning']}\n{indiv['prompt_function']}\n"

            prompt_content = f"{self.task}\n"\
                f"I have {len(indivs)} existing prompt generation approaches with their functions as follows:\n"\
                f"{prompt_indiv}"\
                "Please help me create a new prompt generation approach that is totally different from the given ones.\n"\
                f"{self._base_prompt(mode)}"
            return prompt_content
        
    def _get_prompt_e2(self, indivs: list, mode: PromptMode):
        if mode == PromptMode.CODE:
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
        elif mode == PromptMode.PROMPT:
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
        elif mode == PromptMode.TOOL:
            raise NotImplementedError

    def _get_prompt_m1(self, indiv: dict, mode: PromptMode):
        if mode == PromptMode.CODE:
            prompt_content = f"{self.task}\n"\
                "I have one algorithm with its code as follows.\n"\
                f"Algorithm description: {indiv['algorithm']}\n"\
                f"Code:\n{indiv['code']}\n"\
                "Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided.\n"\
                f"{self._base_prompt(mode)}"
            return prompt_content
        elif mode == PromptMode.PROMPT:
            prompt_content = f"{self.task}\n"\
                "I have one prompt generation approach with its function as follows.\n"\
                f"Reasoning: {indiv['reasoning']}\n"\
                f"Function:\n{indiv['prompt_function']}\n"\
                "Please assist me in creating a new prompt generation approach that has a different form but can be a modified version of the approach provided.\n"\
                f"{self._base_prompt(mode)}"
            return prompt_content
        elif mode == PromptMode.TOOL:
            raise NotImplementedError

    def _get_prompt_m2(self, indiv: dict, mode: PromptMode):
        if mode == PromptMode.CODE:
            prompt_content = f"{self.task}\n"\
                "I have one algorithm with its code as follows.\n"\
                f"Algorithm description: {indiv['algorithm']}\n"\
                f"Code:\n{indiv['code']}\n"\
                "Please identify the main algorithm parameters and assist me in creating a new algorithm that has different parameter settings of the score function provided.\n"\
                f"{self._base_prompt(mode)}"
            return prompt_content
        elif mode == PromptMode.PROMPT:
            prompt_content = f"{self.task}\n"\
                "I have one prompt generation approach with its function as follows.\n"\
                f"Reasoning: {indiv['reasoning']}\n"\
                f"Function:\n{indiv['prompt_function']}\n"\
                "Please identify the main approach parameters and assist me in creating a new prompt generation approach that has different parameter settings of the prompt generation function provided.\n"\
                f"{self._base_prompt(mode)}"
            return prompt_content
        elif mode == PromptMode.TOOL:
            raise NotImplementedError
    
        
        
class Node(ABC):
    def __init__(self, meta_prompt: MetaPrompt):
        self.meta_prompt = meta_prompt
    
    @abstractmethod
    def evolve(self):
        pass
    

class CodeNode(Node):
    def __init__(self, code, algorithm):
        self.code = code
        self.algorithm = algorithm

    def evolve(self):
        # Implementation for evolving code
        pass

class PromptNode(Node):
    def __init__(self, prompt):
        self.prompt = prompt

    def evolve(self):
        # Implementation for evolving prompts
        pass


class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, node1, node2):
        self.edges.append((node1, node2))



# Evolution: 
# - (Hyper-parameter) Meta-Heuristic: Task, FuncName, Input, Output
# - (get_prompt_xx) Evolution tactic specific prompt templating, used for generating node
# - (_get_node) Get prompt response from LLM, parse node content (code, tool, prompt) 
# - (xx) i1/e1/m1/m2/e2 Evolution search on node
#
# Missing: Topology search of our graph

class Evolution:
    def __init__(self, api_endpoint, api_key, model_LLM, llm_use_local, llm_local_url, debug_mode, prompts, **kwargs):
        # ... (keep the existing initialization code)
        self.graph = Graph()

    def evolve_graph(self):
        for node in self.graph.nodes:
            node.evolve()

    # ... (keep other existing methods, but modify them to work with the graph structure)

    def _get_alg(self, prompt_content):
        # ... (existing implementation)
        code_node = CodeNode(code_all, algorithm)
        self.graph.add_node(code_node)
        return code_node

    # Modify i1, e1, e2, m1, m2 methods to work with the graph structure
    def i1(self):
        prompt_content = self.get_prompt_i1()
        prompt_node = PromptNode(prompt_content)
        self.graph.add_node(prompt_node)
        
        code_node = self._get_alg(prompt_content)
        self.graph.add_edge(prompt_node, code_node)
        
        return code_node