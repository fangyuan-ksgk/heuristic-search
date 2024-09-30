from abc import ABC, abstractmethod
from .meta_prompt import MetaPrompt, PromptMode, parse_evol_response
from .meta_prompt import MetaPlan, extract_json_from_text
from .meta_execute import call_func_code, call_func_prompt
from .llm import get_openai_response as get_response
import re, os, json
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


    def _evolve(self, method: str, parents: list = None, replace=False):
        """
        Note: Evolution process will be decoupled with the fitness assignment process
        """
        prompt_method = getattr(self.meta_prompt, f'_get_prompt_{method}')
        prompt_content = prompt_method()
     
        response = get_response(prompt_content)
        reasoning, code = parse_evol_response(response)
        
        if replace:
            self.reasoning, self.code = reasoning, code
        return reasoning, code
    

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
    
    def __call__(self, inputs):
        """ 
        TBD: Inheritance to accumulated codebase with 'file_path' | Graph Topology naturally enables inheritance
        TBD: Stricter input / output type checking to ensure composibility
        """
        if self.meta_prompt.mode == PromptMode.CODE:
            output_value = call_func_code(inputs, self.code, self.meta_prompt.func_name, file_path=None)
            output_name = self.meta_prompt.output[0]
            return {output_name: output_value}
        elif self.meta_prompt.mode == PromptMode.PROMPT:
            output_value = call_func_prompt(inputs, self.code, get_response)
            output_name = self.meta_prompt.output[0]
            return {output_name: output_value}
        
    def save(self, node_path: str) -> None:
        node_data = {
            "code": self.code,
            "reasoning": self.reasoning,
            "meta_prompt": self.meta_prompt.to_dict()  # Assuming MetaPrompt has a to_dict method
        }
        os.makedirs(os.path.dirname(node_path), exist_ok=True)
        with open(node_path, 'w') as f:
            json.dump(node_data, f, indent=2)

    @classmethod 
    def load(cls, node_path: str) -> 'EvolNode':
        with open(node_path, 'r') as f:
            node_data = json.load(f)
        meta_prompt = MetaPrompt.from_dict(node_data['meta_prompt'])  # Assuming MetaPrompt has a from_dict method
        return cls(meta_prompt=meta_prompt, code=node_data['code'], reasoning=node_data['reasoning'])
     


class EvolGraph:
    """ 
    EvolGraph: DNA -- EvolNode: 
    """
    def __init__(self, nodes: Optional[List[EvolNode]], edges: Optional[List], eval_node: Optional[EvolNode]):
        self.nodes: Dict[str, EvolNode] = nodes
        self.edges = edges
        self.eval_node = eval_node
    
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
        
        prompt_content = meta_prompt._get_eval_prompt_i1()
        response = get_response(prompt_content)
        plan_dict = extract_json_from_text(response)
        eval_prompt = MetaPrompt(
            task=node.get("task"),
            func_name=node.get("name"),
            input=node.get("input"),
            output=node.get("output"),
            mode=node.get("mode").lower()
        )
        eval_node = EvolNode(meta_prompt=eval_prompt)
        
        graph = cls(nodes=list(nodes.values()), edges=edges, eval_node=eval_node)

        return graph
    
    def evolve_graph(self, method: str):
        """ 
        Evolve Planning Graph Topology
        """ 
        raise NotImplementedError

    def evolve_node(self, node_id: str, method: str):
        node = self.get_node(node_id)
        if node:
            input_nodes = self.get_input_nodes(node_id)
            if method == 'i1':
                return node.i1()
            elif method in ['e1', 'e2', 'm1', 'm2']:
                return getattr(node, method)(input_nodes)
        return None
    
    def save(self, graph_path: str) -> None:
        graph_data = {
            "nodes": {name: node.save(os.path.join(graph_path, f"node_{name}.json")) 
                      for name, node in self.nodes.items()},
            "edges": self.edges,
            "eval_node": self.eval_node.save(os.path.join(graph_path, "eval_node.json")) if self.eval_node else None
        }
        os.makedirs(graph_path, exist_ok=True)
        with open(os.path.join(graph_path, "graph_structure.json"), 'w') as f:
            json.dump(graph_data, f, indent=2)

    @classmethod
    def load(cls, graph_path: str) -> 'EvolGraph':
        with open(os.path.join(graph_path, "graph_structure.json"), 'r') as f:
            graph_data = json.load(f)
        
        nodes = {name: EvolNode.load(os.path.join(graph_path, f"node_{name}.json")) 
                 for name in graph_data["nodes"]}
        edges = graph_data["edges"]
        eval_node = EvolNode.load(os.path.join(graph_path, "eval_node.json")) if graph_data["eval_node"] else None
        
        return cls(nodes=nodes, edges=edges, eval_node=eval_node)