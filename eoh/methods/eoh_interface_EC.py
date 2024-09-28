import numpy as np
from .eoh_evolution import EvolGraph, EvolNode
from joblib import Parallel, delayed
import concurrent.futures

class InterfaceEC():
    def __init__(self, pop_size: int, m: int,
                 goal: str,
                 select,  # Special selection class
                 n_p: int, timeout: int, use_numba: bool, **kwargs):
        
        self.pop_size = pop_size
        self.eval = EvolNode
        # self.interface_eval = interface_prob # Manual Efforts required for curating the evaluation mechanism
        self.evol = EvolGraph.generate(goal)
        self.m = m
        self.select = select
        self.n_p = n_p
        self.timeout = timeout
        self.use_numba = use_numba

    def check_duplicate(self, population, code):
        return any(code == ind['code'] for ind in population)

    def _get_alg(self, pop, operator):
        """ 
        Generate one offspring using specific operator 
        - Select parent
        - Evolve (crossover or mutation)
        """
        offspring = {'algorithm': None, 'code': None, 'objective': None, 'other_inf': None}
        parents = self.select.parent_selection(pop, self.m) if operator != "i1" else None
        
        if operator == "i1":
            offspring['code'], offspring['algorithm'] = self.evol.i1()
        elif operator in ["e1", "e2", "m1", "m2"]:
            method = getattr(self.evol, operator)
            offspring['code'], offspring['algorithm'] = method(parents)
        
        return parents, offspring

    def get_offspring(self, pop, operator):
        """ 
        Generate a unique offspring using specific operator
        - Wrap around _get_alg()
        - Check against duplication
        - Run evalution with timeout handler
        """
        p, offspring = self._get_alg(pop, operator)
        
        while self.check_duplicate(pop, offspring['code']): # Desperate move for diversity ensurance | External stimulus is necessary
            p, offspring = self._get_alg(pop, operator)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.interface_eval.evaluate, offspring['code'])
            fitness = future.result(timeout=self.timeout, default=0.) # Run Evaluation with TimeOut Default score as 0.
            offspring['objective'] = np.round(fitness, 5)

        return p, offspring

    def get_algorithm(self, pop, operator): # Parallel Genetic Search
        """ 
        Generate pop_size unique offsprings using specific operator
        - Parallel execution
        - Wrap around get_offspring()
        """
        results = Parallel(n_jobs=self.n_p, timeout=self.timeout+15)(
            delayed(self.get_offspring)(pop, operator) for _ in range(self.pop_size)
        )
        return zip(*results)