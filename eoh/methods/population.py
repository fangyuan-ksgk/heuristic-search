import os
import json
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

def parent_selection(pop: List[EvolNode], m: int, proportion: float = 0.8) -> List[dict]:
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
    tournament_size = min(len(pop) // m, int(len(pop) * proportion)) # Dynamic tournament size 

    for _ in range(m):
        # Each tournament now looks at population_size/m individuals
        tournament = np.random.choice(pop, size=max(2, tournament_size), replace=False)
        winner = max(tournament, key=lambda x: x['fitness'] if x['fitness'] is not None else float('-inf'))
        selected_parents.append(winner)
    
    return selected_parents


# Evolution process will keep populations of EvolNodes
# - I see the point now, we don't need to use multiple EvolNode for its evolution process, the MetaPromp is designed to incorporate all required information already ...

class Evolution: 
    def __init__(self, pop_size: int, meta_prompt: MetaPrompt, get_response: Callable, test_cases: Optional[list] = None, 
                 max_attempts: int = 3, num_eval_runs: int = 1, num_parents: int = 2, filename: str = "default", load: bool = False): 
        self.pop_size = pop_size
        self.meta_prompt = meta_prompt
        self.evol = EvolNode(meta_prompt, None, None, get_response=get_response, test_cases=test_cases)
        self.max_attempts = max_attempts
        self.num_eval_runs = num_eval_runs
        self.num_parents = num_parents
        if load:
            self.load_population(filename)
        
    def check_duplicate(self, population, code):
        return any(code == ind['code'] for ind in population)
            
    def _get_offspring(self, operator, pop: list = []):
        """ 
        Generate one offspring using specific operator 
        - Select parent
        - Evolve (crossover or mutation)
        """
        
        offspring = {"reasoning": None, "code": None, "fitness": None}
                
        if operator == "i1": # initialization operator 
            self.evol.evolve("i1", replace=True, max_attempts=self.max_attempts, num_runs=self.num_eval_runs)
            offspring["reasoning"], offspring["code"], offspring["fitness"] = self.evol.reasoning, self.evol.code, self.evol.fitness
        elif operator.startswith("e"): # cross-over operator
            parents = parent_selection(pop, self.num_parents) # in fact we don't mind 3P
            self.evol.evolve(operator, parents, replace=True, max_attempts=self.max_attempts, num_runs=self.num_eval_runs)
            offspring["reasoning"], offspring["code"], offspring["fitness"] = self.evol.reasoning, self.evol.code, self.evol.fitness
        elif operator.startswith("m"): # mutation operator
            parents = parent_selection(pop, 1) # one parent used for mutation
            self.evol.evolve(operator, parents[0], replace=True, max_attempts=self.max_attempts, num_runs=self.num_eval_runs)
            offspring["reasoning"], offspring["code"], offspring["fitness"] = self.evol.reasoning, self.evol.code, self.evol.fitness
         
        # add offspring to population
        if not self.check_duplicate(pop, offspring["code"]):
            pop.append(offspring)
            
        return pop
    
    def save_population(self, pop: list, filename: str = "default", population_dir: str = "methods/population"):
        population_folder = f"{population_dir}/{self.meta_prompt.func_name}/"
        os.makedirs(population_folder, exist_ok=True)
        filepath = os.path.join(population_folder, filename)
        with open(filepath, 'w') as f:
            json.dump(pop, f, indent=2)
    
    def load_population(self, filename: str = "default", population_dir: str = "methods/population"):
        filepath = os.path.join(population_dir, filename)
        with open(filepath, 'r') as f:
            pop = json.load(f)
        return pop