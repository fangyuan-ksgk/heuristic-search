import numpy as np
from .evolnode import EvolNode
from .meta_prompt import MetaPrompt
from typing import Callable
# from joblib import Parallel, delayed
from typing import List, Optional

# Managing population of evolving nodes 
# - Natural selection 
# - Offspring generation

# Ideally 

def parent_selection(pop: List[EvolNode], m): 
    raise NotImplementedError

# Evolution process will keep populations of EvolNodes
# - I see the point now, we don't need to use multiple EvolNode for its evolution process, the MetaPromp is designed to incorporate all required information already ...

class Evolution: 
    def __init__(self, pop_size: int, meta_prompt: MetaPrompt, get_response: Callable, test_cases: Optional[list] = None, max_attempts: int = 3, num_eval_runs: int = 1): 
        self.pop_size = pop_size
        self.meta_prompt = meta_prompt
        self.evol = EvolNode(meta_prompt, None, None, get_response=get_response)
        self.max_attempts = max_attempts
        self.num_eval_runs = num_eval_runs
        
    def check_duplicate(self, population, code):
        return any(code == ind['code'] for ind in population)
            
    def _get_offspring(self, pop, operator):
        """ 
        Generate one offspring using specific operator 
        - Select parent
        - Evolve (crossover or mutation)
        """
        
        offspring = {"reasoning": None, "code": None, "goal": None, "fitness": None}
        parents = parent_selection(pop, self.m) if operator != "i1" else None
        
        if operator == "i1":
            self.evol.i1(replace=True, max_attempts=self.max_attempts, num_runs=self.num_eval_runs)
            offspring["reasoning"], offspring["code"], offspring["goal"], offspring["fitness"] = self.evol.reasoning, self.evol.code, self.evol.goal, self.evol.fitness
        else: 
            get_method = getattr(self.evol, operator)
            get_method(parents)
            offspring["reasoning"], offspring["code"], offspring["goal"], offspring["fitness"] = self.evol.reasoning, self.evol.code, self.evol.goal, self.evol.fitness
         
        # Issue: for crossover, we need to inform operator of other parent's information, so that operation should not be in EvolNode class (?)