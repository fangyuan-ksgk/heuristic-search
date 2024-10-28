import json
import random
from .population import InterfaceEC

class EOH:
    def __init__(self, paras, problem, select, manage):
        self.prob = problem
        self.select = select
        self.manage = manage
        
        self.pop_size = paras.ec_pop_size
        self.n_pop = paras.ec_n_pop
        self.operators = paras.ec_operators
        self.operator_weights = paras.ec_operator_weights
        
        self.output_path = paras.exp_output_path

    def add2pop(self, population, offspring):
        for off in offspring:
            if not any(ind['objective'] == off['objective'] for ind in population):
                population.append(off)

    def run(self):
        interface_ec = InterfaceEC(self.prob, self.select)
        population = interface_ec.population_generation()

        for pop in range(self.n_pop):
            for i, op in enumerate(self.operators):
                if random.random() < self.operator_weights[i]:
                    parents, offsprings = interface_ec.get_algorithm(population, op)
                    self.add2pop(population, offsprings)
                
                population = self.manage.population_management(population, self.pop_size)

            self.save_population(population, pop)

    def save_population(self, population, generation):
        filename = f"{self.output_path}/population_generation_{generation}.json"
        with open(filename, 'w') as f:
            json.dump(population, f, indent=2)