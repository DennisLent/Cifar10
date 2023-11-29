import processing
import os
from tensorflow.keras.utils import to_categorical
from population import Population
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load and preprocess the CIFAR-10 dataset
(train_images, train_labels, _) = processing.create_data("cifar-10-batches-py", "data")
(test_images, test_labels, _) = processing.create_data("cifar-10-batches-py", "test")

#normalize
train_images = train_images / 255
test_images = test_images / 255

train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

train = (train_images, train_labels)
test = (test_images, test_labels)

def run_algorithm(population_size: int, max_runs: int, max_epochs: int, seed: int = None, verbose: bool = False):
    """
    Main function to run the algorithm
    """
    evolution = {"Generation": [], "Fitness": []}

    gen = 0
    population = Population()
    population.initialize(population_size, seed)

    while gen < max_runs:

        #evaluate the fitness of the population
        population.fitness(train, test, max_epochs, verbose)

        #best fitness of the generation
        best_fitness = max(population.individuals, key=lambda x: x.fitness_score).fitness_score

        evolution["Generation"].append(gen)
        evolution["Fitness"].append(best_fitness)

        print(f"Generation: {gen} | Best Fitness: {best_fitness}")

        #next generation
        population.new_generation()
        gen += 1

