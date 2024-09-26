from abc import ABC, abstractmethod
from .meta_prompt import MetaPrompt, PromptMode, parse_evol_response
from .meta_prompt import MetaPlan, extract_json_from_text
from .llm import get_openai_response as get_response
import re
from typing import Optional, Dict, List

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
        
class EvolNode:
    
    def __init__(self, meta_prompt: MetaPrompt, code: Optional[str] = None, reasoning: Optional[str] = None):
        self.code = code
        self.reasoning = reasoning
        self.meta_prompt = meta_prompt
    
    def _evolve(self, method: str, parents: list = None):
        prompt_method = getattr(self.meta_prompt, f'_get_prompt_{method}')
        prompt_content = prompt_method()
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
    EvolGraph: DNA -- EvolNode: 
    """
    def __init__(self, nodes: Optional[List[EvolNode]], edges: Optional[List]):
        self.nodes: Dict[str, EvolNode] = {}
        self.edges = []
    
    @classmethod
    def generate(cls, goal: str):
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
                input=node.get("input"),
                output=node.get("output"),
                mode=node.get("mode").lower()
            )
            nodes[node.get("name")] = EvolNode(meta_prompt=node_prompt)
        
        edges = plan_dict["edges"]
        
        graph = cls(nodes=list(nodes.values()), edges=edges)
        graph.nodes = nodes
        graph.edges = edges
        
        return graph

    def evolve_node(self, node_id: str, method: str):
        node = self.get_node(node_id)
        if node:
            input_nodes = self.get_input_nodes(node_id)
            if method == 'i1':
                return node.i1()
            elif method in ['e1', 'e2', 'm1', 'm2']:
                return getattr(node, method)(input_nodes)
        return None