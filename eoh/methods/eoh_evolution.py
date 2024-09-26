from abc import ABC, abstractmethod
from .meta_prompt import MetaPrompt, PromptMode
import re

#################################
#   Evolution on Graph          #
#   - Node: Code, Prompt, Tool  #
#   - Edge: Not Decided Yet     #
#################################
        
class Node(ABC):
    def __init__(self, content: str, description: str, meta_prompt: MetaPrompt):
        self.content = content
        self.description = description
        self.meta_prompt = meta_prompt
        self.prompt_mode: PromptMode = PromptMode.PROMPT
    
    @abstractmethod
    def evolve(self):
        pass

class CodeNode(Node):
    def __init__(self, code: str, description: str, meta_prompt: MetaPrompt):
        super().__init__(code, description, meta_prompt)
        self.prompt_mode = PromptMode.CODE

    def evolve(self):
        # Implementation for evolving code
        # Use self.content (code), self.description, and self.meta_prompt
        pass

class PromptNode(Node):
    def __init__(self, prompt: str, description: str, meta_prompt: MetaPrompt):
        super().__init__(prompt, description, meta_prompt)
        self.prompt_mode = PromptMode.PROMPT

    def evolve(self):
        # Implementation for evolving prompts
        # Use self.content (prompt), self.description, and self.meta_prompt
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
    
    
    def _get_alg(self,prompt_content):
        """ 
        Compile Prompt with LLM
        Parsing code from response
        """

        response = self.interface_llm.get_response(prompt_content)

        algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
        if len(algorithm) == 0:
            if 'python' in response:
                algorithm = re.findall(r'^.*?(?=python)', response,re.DOTALL)
            elif 'import' in response:
                algorithm = re.findall(r'^.*?(?=import)', response,re.DOTALL)
            else:
                algorithm = re.findall(r'^.*?(?=def)', response,re.DOTALL)

        code = re.findall(r"import.*return", response, re.DOTALL)
        if len(code) == 0:
            code = re.findall(r"def.*return", response, re.DOTALL)

        n_retry = 1
        while (len(algorithm) == 0 or len(code) == 0):
            if self.debug_mode:
                print("Error: algorithm or code not identified, wait 1 seconds and retrying ... ")

            response = self.interface_llm.get_response(prompt_content)

            algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
            if len(algorithm) == 0:
                if 'python' in response:
                    algorithm = re.findall(r'^.*?(?=python)', response,re.DOTALL)
                elif 'import' in response:
                    algorithm = re.findall(r'^.*?(?=import)', response,re.DOTALL)
                else:
                    algorithm = re.findall(r'^.*?(?=def)', response,re.DOTALL)

            code = re.findall(r"import.*return", response, re.DOTALL)
            if len(code) == 0:
                code = re.findall(r"def.*return", response, re.DOTALL)
                
            if n_retry > 3:
                break
            n_retry +=1

        algorithm = algorithm[0]
        code = code[0] 

        code_all = code+" "+", ".join(s for s in self.prompt_func_outputs) 


        return [code_all, algorithm]

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