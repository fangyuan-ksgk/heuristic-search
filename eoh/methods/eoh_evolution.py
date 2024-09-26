from abc import ABC, abstractmethod
from .meta_prompt import MetaPrompt

#################################
#   Evolution on Graph          #
#   - Node: Code, Prompt, Tool  #
#   - Edge: Not Decided Yet     #
#################################
        
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