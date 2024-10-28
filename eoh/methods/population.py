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

def parent_selection(pop: List[EvolNode], m: int) -> List[dict]:
    """
    Select m parents from the population using tournament selection.
    
    Args:
        pop: List of dictionaries containing individuals with their properties
        m: Number of parents to select
        tournament_size: Number of individuals to compete in each tournament
    
    Returns:
        List of selected parent dictionaries
    """
    selected_parents = []
    tournament_size = len(pop) // m # Dynamic tournament size!

    for _ in range(m):
        # Each tournament now looks at population_size/m individuals
        tournament = np.random.choice(pop, size=max(2, tournament_size), replace=False)
        winner = max(tournament, key=lambda x: x['fitness'] if x['fitness'] is not None else float('-inf'))
        selected_parents.append(winner)
    
    return selected_parents


# Evolution process will keep populations of EvolNodes
# - I see the point now, we don't need to use multiple EvolNode for its evolution process, the MetaPromp is designed to incorporate all required information already ...

class Evolution: 
    def __init__(self, pop_size: int, meta_prompt: MetaPrompt, get_response: Callable, test_cases: Optional[list] = None, max_attempts: int = 3, num_eval_runs: int = 1): 
        self.pop_size = pop_size
        self.meta_prompt = meta_prompt
        self.evol = EvolNode(meta_prompt, None, None, get_response=get_response, test_cases=test_cases)
        self.max_attempts = max_attempts
        self.num_eval_runs = num_eval_runs
        
    def check_duplicate(self, population, code):
        return any(code == ind['code'] for ind in population)
            
    def _get_offspring(self, operator, pop: list = []):
        """ 
        Generate one offspring using specific operator 
        - Select parent
        - Evolve (crossover or mutation)
        """
        
        offspring = {"reasoning": None, "code": None, "fitness": None}
        parents = parent_selection(pop, self.m) if operator != "i1" else None
                
        if operator == "i1":
            self.evol.evolve("i1", replace=True, max_attempts=self.max_attempts, num_runs=self.num_eval_runs)
            offspring["reasoning"], offspring["code"], offspring["fitness"] = self.evol.reasoning, self.evol.code, self.evol.fitness
        else: 
            self.evol.evolve(operator, parents, replace=True, max_attempts=self.max_attempts, num_runs=self.num_eval_runs)
            offspring["reasoning"], offspring["code"], offspring["fitness"] = self.evol.reasoning, self.evol.code, self.evol.fitness
         
        # add offspring to population
        if not self.check_duplicate(pop, offspring["code"]):
            pop.append(offspring)
            
        return pop