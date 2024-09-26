from abc import ABC, abstractmethod
from .meta_prompt import MetaPrompt, PromptMode, parse_evol_response
from .llm import get_openai_response as get_response
import re

#################################
#   Evolution on Graph          #
#   - Node: Code, Prompt, Tool  #
#   - Edge: Not Decided Yet     #
#################################
        
class EvolNode(ABC):
    def __init__(self, content: str, description: str, meta_prompt: MetaPrompt):
        self.content = content
        self.description = description
        self.meta_prompt = meta_prompt
        self.prompt_mode: PromptMode = PromptMode.PROMPT
    
    def _evolve(self, method: str, parents: list = None):
        prompt_method = getattr(self.meta_prompt, f'_get_prompt_{method}')
        prompt_args = [self.prompt_mode] if parents is None else [parents, self.prompt_mode]
        prompt_content = prompt_method(*prompt_args)
        response = get_response(prompt_content)
        return parse_evol_response(response)

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
    def __init__(self, prompts, **kwargs):
        # ... (keep the existing initialization code)
        self.graph = Graph()

    def evolve_graph(self):
        for node in self.graph.nodes:
            node.evolve()

    # ... (keep other existing methods, but modify them to work with the graph structure)
    
    
    def _get_node(self, prompt_content):
        """ 
        Compile Prompt with LLM
        Parsing code from response
        """

        response = get_response(prompt_content)

        reasoning, code = parse_node(response)

        n_retry = 1
        while (len(reasoning) == 0 or len(code) == 0):
            if self.debug_mode:
                print("Error: reasoning or code not identified, wait 1 seconds and retrying ... ")

            response = get_response(prompt_content)
            reasoning, code = parse_node(response)
                
            if n_retry > 3:
                break
            n_retry +=1

        reasoning = reasoning[0]
        code = code[0] 

        code_all = code+" "+", ".join(s for s in self.prompt_func_outputs) 

        return [code_all, reasoning]

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