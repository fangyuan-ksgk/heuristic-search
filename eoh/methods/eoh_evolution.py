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


# Evolution: 
# - (Hyper-parameter) Meta-Heuristic: Task, FuncName, Input, Output
# - (get_prompt_xx) Evolution tactic specific prompt templating, used for generating node
# - (_get_node) Get prompt response from LLM, parse node content (code, tool, prompt) 
# - (xx) i1/e1/m1/m2/e2 Evolution search on node
#
# Missing: Topology search of our graph
        
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

class EvolGraph:
    """ 
    Perhaps we should fold everything about 'Evolution' into this EvolGraph class
    - Deciding what happen when 'evol' is called etc.
    """
    def __init__(self):
        self.nodes = {}  # Use a dictionary to store nodes with unique identifiers
        self.edges = []

    def add_node(self, node_id: str, node: EvolNode):
        self.nodes[node_id] = node

    def add_edge(self, from_node_id: str, to_node_id: str):
        if from_node_id in self.nodes and to_node_id in self.nodes:
            self.edges.append((from_node_id, to_node_id))
        else:
            raise ValueError("Both nodes must exist in the graph before adding an edge.")

    def get_node(self, node_id: str) -> Optional[EvolNode]:
        return self.nodes.get(node_id)

    def get_input_nodes(self, node_id: str) -> list[EvolNode]:
        return [self.nodes[from_id] for from_id, to_id in self.edges if to_id == node_id]

    def get_output_nodes(self, node_id: str) -> list[EvolNode]:
        return [self.nodes[to_id] for from_id, to_id in self.edges if from_id == node_id]
    
    @classmethod
    def parse_graph_from_response(cls, response):
        pass

    def evolve_node(self, node_id: str, method: str):
        node = self.get_node(node_id)
        if node:
            input_nodes = self.get_input_nodes(node_id)
            if method == 'i1':
                return node.i1()
            elif method in ['e1', 'e2', 'm1', 'm2']:
                return getattr(node, method)(input_nodes)
        return None