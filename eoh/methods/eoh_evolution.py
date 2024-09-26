from abc import ABC, abstractmethod
from .meta_prompt import MetaPrompt, PromptMode, parse_evol_response
from .llm import get_openai_response as get_response
import re
from typing import Optional

#################################
#   Evolution on Graph          #
#   - Node: Code, Prompt, Tool  #
#   - Edge: Not Decided Yet     #
#################################
        
class EvolNode(ABC):
    
    def __init__(self, code: Optional[str], reasoning: Optional[str], meta_prompt: MetaPrompt, prompt_mode: PromptMode):
        self.code = code
        self.reasoning = reasoning
        self.meta_prompt = meta_prompt
        self.prompt_mode = prompt_mode
    
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



# Evolution: 
# - (Hyper-parameter) Meta-Heuristic: Task, FuncName, Input, Output
# - (get_prompt_xx) Evolution tactic specific prompt templating, used for generating node
# - (_get_node) Get prompt response from LLM, parse node content (code, tool, prompt) 
# - (xx) i1/e1/m1/m2/e2 Evolution search on node
#
# Missing: Topology search of our graph


class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, node1, node2):
        self.edges.append((node1, node2))