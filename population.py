import numpy as np
from model import create_model, sim_model

# there will be 3 convolutional layers and 2 fully connected layers
conv_layers = 3
f_layers = 2

class Individual:
    def __init__(self) -> None:
        #genes as follows [conv1, conv2, conv3, s_conv1, s_conv2, s_conv3, s_fc1, s_fc2]
        self.genes = []
    
    def initialize(self, seed=None):
        if seed:
            np.random.seed(seed)
        else:
            np.random.seed(204)
        
        # initialize weights randomly
        for _ in range(conv_layers):
            self.genes.append(np.random.randint(10,128))
        for _ in range(conv_layers, 2*conv_layers):
            self.genes.append(np.random.randint(2, 4))
        for _ in range(2*conv_layers, conv_layers + f_layers):
            self.genes.append(np.random.randint(10, 1024))

class Population:
    def __init__(self) -> None:
        self.individuals = []
        self.fitness = []
    
    #initialize population
    def initialize(self, popsize: int, seed=None):
        if popsize % 2 == 0:
            for _ in range(popsize):
                ind = Individual.initialize(seed)
                self.population.append(ind)
        else:
            print("Population has to be a multiple of 2")
    
    def fitness(self, train, test, epochs):
        for individual in self.individuals:
            model = create_model(individual)
            acc = sim_model(model, train, test, epochs)
            self.fitness.append(acc)
        
