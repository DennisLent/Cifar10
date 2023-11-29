import numpy as np
from model import create_model, sim_model

# there will be 3 convolutional layers and 2 fully connected layers
conv_layers = 3
f_layers = 2

class Individual:
    def __init__(self) -> None:
        #genes as follows [conv1, conv2, conv3, s_conv1, s_conv2, s_conv3, s_fc1, s_fc2]
        self.genes = []
        self.fitness_score = 0
    
    def initialize(self, seed=None):
        if seed:
            np.random.seed(seed)
        else:
            np.random.seed(0)
        
        # initialize weights randomly
        for _ in range(conv_layers):
            self.genes.append(np.random.randint(10,128))
        for _ in range(conv_layers, 2*conv_layers):
            self.genes.append(np.random.randint(2, 6))
        for _ in range(2*conv_layers, 2*conv_layers + f_layers):
            self.genes.append(np.random.randint(10, 1024))
    
    def mutate(self):
        mutation_index = np.random.randint(len(self.genes))
        if mutation_index < conv_layers:
            self.genes[mutation_index] = np.random.randint(10, 128)
        elif conv_layers < mutation_index < 2 * conv_layers:
            self.genes[mutation_index] = np.random.randint(2, 6)
        else:
            self.genes[mutation_index] = np.random.randint(10, 1024)
    
    @staticmethod
    def random_crossover(individual1, individual2):
        crossover_point = np.random.randint(len(individual1.genes))
        offspring1 = Individual()
        offspring2 = Individual()
        offspring1.genes = individual1.genes[:crossover_point] + individual2.genes[crossover_point:]
        offspring2.genes = individual2.genes[:crossover_point] + individual1.genes[crossover_point:]
        return (offspring1, offspring2)

class Population:
    def __init__(self) -> None:
        self.individuals = []
    
    #initialize population
    def initialize(self, popsize: int, seed=None):
        print("-Initializing Population")
        if popsize % 2 == 0:
            for _ in range(popsize):
                ind = Individual()
                ind.initialize(seed)
                self.individuals.append(ind)
        else:
            print("Population has to be a multiple of 2")
    
    def fitness(self, train, test, epochs, verbose=False):
        print("--Evaluating Fitness")
        if verbose:
            index = 0
        for individual in self.individuals:
            if verbose:
                print(f"---Evaluating Individual {index}")
                index +=1
            model = create_model(individual)
            individual.fitness_score = sim_model(model, train, test, epochs)
    
    def new_generation(self):
        print("--Creating New Generation")
        best = self.individuals.sort(key=lambda x: x.fitness_score, reverse=True)

        # use the top 25% for next generation to preserve some genes
        new_generation = best[:len(self.individuals)//4]

        #fill the rest of the new generation with offspring
        while len(new_generation) < len(self.individuals):
            # don't replace
            parent1, parent2 = np.random.choice(self.individuals, size=2, replace=False)

            child1, child2 = Individual.random_crossover(parent1, parent2)

            #random chance to mutate a child
            c1, c2 = np.random.random(), np.random.random()
            if c1 < 0.15:
                child1.mutate()
            if c2 < 0.15:
                child2.mutate()
            
            new_generation.append(child1)
            new_generation.append(child2)
        
        self.individuals = new_generation
        
        
