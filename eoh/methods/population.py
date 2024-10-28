import numpy as np
from .evolnode import EvolGraph, EvolNode
from joblib import Parallel, delayed
import concurrent.futures
from typing import List

# Managing population of evolving nodes 
# - Natural selection 
# - Offspring generation

# Ideally 

def parent_selection(pop: List[EvolNode], m): 
    raise NotImplementedError

# Evolution process will keep populations of EvolNodes
class Evolution: 
    def __init__(self, pop_size: int, goal: str): 
        self.pop_size = pop_size
        self.goal = goal
        self.evol = None 
        
    def check_duplicate(self, population, code):
        return any(code == ind['code'] for ind in population)
            
    def _get_offspring(self, pop, operator):
        """ 
        Generate one offspring using specific operator 
        - Select parent
        - Evolve (crossover or mutation)
        """
        offspring = {'algorithm': None, 'code': None, 'objective': None, 'other_inf': None}
        parents = parent_selection(pop, self.m) if operator != "i1" else None
        
        if operator == "i1":
            offspring['code'], offspring['algorithm'] = self.evol.i1()
        else: 
             raise NotImplementedError
         
        # Issue: for crossover, we need to inform operator of other parent's information, so that operation should not be in EvolNode class (?)