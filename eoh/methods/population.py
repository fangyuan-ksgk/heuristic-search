import os
import json
import numpy as np
from .evolnode import EvolNode
from .meta_prompt import MetaPrompt, PromptMode
from typing import Callable
# from joblib import Parallel, delayed
from typing import List, Optional, Union
from collections import defaultdict
import random
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

    while len(selected_parents) != m:
        # Each tournament now looks at population_size/m individuals
        tournament = np.random.choice(pop, size=max(2, tournament_size), replace=False)
        winner = max(tournament, key=lambda x: x['fitness'] if x['fitness'] is not None else float('-inf'))
        if winner not in selected_parents:
            selected_parents.append(winner)
    
    return selected_parents


# Evolution process will keep populations of EvolNodes
# - I see the point now, we don't need to use multiple EvolNode for its evolution process, the MetaPromp is designed to incorporate all required information already ...


def indiv_to_prompt(indiv: dict, mode: PromptMode) -> str:
    if mode == PromptMode.PROMPT:
        prompt_indiv = f"[APPROACH]: {indiv['reasoning']}\n[PROMPT FUNCTION]: {indiv['code']}\n[FITNESS]: {indiv['fitness']}\n"
    elif mode == PromptMode.CODE:
        prompt_indiv = f"[ALGORITHM]: {indiv['reasoning']}\n[CODE]: {indiv['code']}\n[FITNESS]: {indiv['fitness']}\n"
    return prompt_indiv

class Evolution: 
    
    def __init__(self, pop_size: int, meta_prompt: MetaPrompt, get_response: Callable, test_cases: Optional[list] = None, 
                 max_attempts: int = 3, num_eval_runs: int = 1, num_parents: int = 2, filename: str = "default", load: bool = False, mutation_rate: float = 0.1): 
        self.pop_size = pop_size # not used (for capping population size I suppose?)
        self.meta_prompt = meta_prompt
        self.evol = EvolNode(meta_prompt, None, None, get_response=get_response, test_cases=test_cases)
        self.max_attempts = max_attempts
        self.num_eval_runs = num_eval_runs
        self.num_parents = num_parents
        self.population = []
        self.load = load
        if load:
            self.population = self.load_population(filename)
        self.mutation_rate = mutation_rate
        self.get_response = get_response
        self.strategy_trace = "Initial Population information: " + self.population_info

    def check_duplicate(self, population, code):
        return any(code == ind['code'] for ind in population)
            
    def _get_offspring(self, operator, pop: list = [], max_attempts: int = -1):
        """ 
        Generate one offspring using specific operator 
        - Select parent
        - Evolve (crossover or mutation)
        """
        if max_attempts == -1:
            max_attempts = self.max_attempts
        
        offspring = {"reasoning": None, "code": None, "fitness": None}
                
        if operator == "i1": # initialization operator 
            self.evol.evolve("i1", replace=True, max_tries=max_attempts, num_runs=self.num_eval_runs, batch_size=self.pop_size)
            offspring["reasoning"], offspring["code"], offspring["fitness"] = self.evol.reasoning, self.evol.code, self.evol.fitness
        
        elif operator.startswith("e"): # cross-over operator
            if not self.load and len(pop) < self.num_parents:
                while len(pop) < self.num_parents:
                    pop.extend(self.evol.evolve("i1", max_tries=1, num_runs=self.num_eval_runs, batch_size=self.pop_size))
            assert len(pop) >= self.num_parents, "Population size is less than the number of parents required for Crossover operator"
            parents = parent_selection(pop, self.num_parents) # in fact we don't mind 3P
            self.evol.evolve(operator, parents, replace=True, max_tries=max_attempts, num_runs=self.num_eval_runs, batch_size=self.pop_size)
            offspring["reasoning"], offspring["code"], offspring["fitness"] = self.evol.reasoning, self.evol.code, self.evol.fitness
        
        elif operator.startswith("m"): # mutation operator
            if not self.load and len(pop) < 1:
                pop = self._get_offspring("i1", pop, 1)
                
            assert len(pop) >= 1, "Population size is less than the number of parents required for Mutation operator"
            parents = parent_selection(pop, 1) # one parent used for mutation
            self.evol.evolve(operator, parents[0], replace=True, max_tries=max_attempts, num_runs=self.num_eval_runs, batch_size=self.pop_size)
            offspring["reasoning"], offspring["code"], offspring["fitness"] = self.evol.reasoning, self.evol.code, self.evol.fitness
        
        elif operator == "t": #traditional GA
            if not self.load and len(pop) < self.pop_size:
                while len(pop) < self.pop_size:
                    pop.extend(self.evol.evolve("i1", max_tries=1, num_runs=self.num_eval_runs, batch_size=self.pop_size))
            assert len(pop) >= self.num_parents, "Population size is less than the number of parents required for Traditional operator"
            for i in range(max_attempts):
                parents = parent_selection(pop, self.num_parents)
                offspring = []
                while len(offspring) == 0 or self.check_duplicate(pop, offspring[0]["code"]):
                    offspring = self.evol.evolve("e1", parents, max_tries=1, num_runs=self.num_eval_runs, batch_size=self.pop_size)
                offspring = offspring[0]
                if random.random() < self.mutation_rate:
                    offspring_m = []
                    while len(offspring_m) == 0 or self.check_duplicate(pop, offspring_m[0]["code"]):
                        offspring_m = self.evol.evolve("m1", offspring, max_tries=1, num_runs=self.num_eval_runs, batch_size=self.pop_size)
                    offspring = offspring_m[0]
                offspring_info = f"Going through {1} of {operator} evolution steps, obtaining offspring:\n {indiv_to_prompt(offspring, self.meta_prompt.mode)}"
                self.strategy_trace += offspring_info

                pop.append(offspring)
                if offspring["fitness"] == 1.0: break
            return pop
        
        # Evolution Info Tracing
        offspring_info = f"Going through {max_attempts} of {operator} evolution steps, obtaining offspring:\n {indiv_to_prompt(offspring, self.meta_prompt.mode)}"
        self.strategy_trace += offspring_info

        # add offspring to population
        if not self.check_duplicate(pop, offspring["code"]):
            pop.append(offspring)
            
        return pop
    
    def get_offspring(self, method: Union[str, List[str]] = "default"):
        if isinstance(method, str):
            self.population = self._get_offspring(method, self.population)
        elif isinstance(method, list):
            for m in method:
                self.population = self._get_offspring(m, self.population)
        
    def save_population(self, pop: list, filename: str = "default", population_dir: str = "methods/population"):
        population_folder = f"{population_dir}/{self.meta_prompt.func_name}/"
        os.makedirs(population_folder, exist_ok=True)
        filepath = os.path.join(population_folder, filename+".json")
        with open(filepath, 'w') as f:
            json.dump(pop, f, indent=2)
            
    @property 
    def population_info(self):
        pop = self.population
        if not pop:
            return "Population is empty"
        if len(pop) == 1:
            indiv = pop[0]
            return f"Population size is 1, information on the best individual:\n {indiv_to_prompt(indiv, self.meta_prompt.mode)}"
        else: 
           # get the best 2 individuals from the population (if don't have 2, use 1)
           best_2 = sorted(pop, key=lambda x: x['fitness'], reverse=True)[:2]
           pop_size = len(pop)
           best_indiv_info = ""
           for i, indiv in enumerate(best_2, 1):
               best_indiv_info += f"Individual {i}:\n{self.meta_prompt._get_prompt_indivs(indiv)}\n"
           return f"Population size: {pop_size}\nBest Fitness: {best_2[0]['fitness']}\nInformation on the best 2 individuals:\n{best_indiv_info}"
    
    def load_population(self, filename: str = "default", population_dir: str = "methods/population"):
        population_folder = f"{population_dir}/{self.meta_prompt.func_name}/"
        filepath = os.path.join(population_folder, filename+".json")
        with open(filepath, 'r') as f:
            pop = json.load(f)
        return pop
    
    def chat(self, question: str = "How effective is the current evolution strategy? What improvement has it made in terms of fitness, and in terms of the implementation?",
             get_response: Optional[Callable] = None):
        prompt = f"{question}\n\n{self.strategy_trace}"
        response = self.get_response(prompt) if get_response is None else get_response(prompt)
        for line in response.split('\n'):
            print(line)
        # return response