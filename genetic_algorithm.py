import numpy as np
import matplotlib.pyplot as plt
import random

# Use autoencoder to decode the genomes
def decode_genome(decoder, genome):
    decoded_image = decoder.predict(genome)
    return decoded_image

# Evaluate the fitness of the decoded images
def fitness_evaluation(decoder, encoded_population, encoded_target_images):
    fitness_values = []
    for genome in encoded_target_images:
        # TODO: compare decoded population and target images?
        # decoded_genome = decode_genome(decoder, genome)
        fitnesses = [calculate_fitness(genome, original_image) for original_image in encoded_population]
        fitness = max(fitnesses)
        fitness_values.append(fitness)
    return np.array(fitness_values)

# Calculate the fitness of the decoded images
def calculate_fitness(decoded_target_image, original_image):
    fitness_values = []
    for target_vector, original_vector in zip(decoded_target_image, original_image):
        target_flat = target_vector.flatten()
        original_flat = original_vector.flatten()
        mean = np.mean((original_flat - target_flat)**2)
        fitness = 1 / (1 + mean)
        fitness_values.append(fitness)
    return np.mean(fitness_values)

# Select the best genomes based on fitness score
def select_parents(population, fitness_scores):
    selected_indices = np.argsort(fitness_scores)[:len(population) // 2]
    selected_population = [population[i] for i in selected_indices]
    return selected_population

# Mutate genomes to create new offspring
def mutate_genome(genome, mutation_rate):
    mutated_genome = genome + np.random.normal(0, mutation_rate, genome.shape)
    return mutated_genome

# Create offspring by crossover and mutation
def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    return child

# Generate new population from parents
def generate_new_population(parents, population_size, mutation_rate):
    new_population = parents.copy()

    while len(new_population) < population_size:
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        child = crossover(parent1, parent2)

        if random.random() < mutation_rate:
            child = mutate_genome(child, mutation_rate)

        new_population.append(child)

    return new_population

# Genetic algorithm function
def genetic_algorithm(decoder, population, victim_choice, population_size, mutation_rate):

    fitness_scores = fitness_evaluation(decoder, population, victim_choice)
    parents = select_parents(population, fitness_scores)
    new_population = generate_new_population(parents, population_size, mutation_rate)

    # lowest_fitness_score_genome = population[np.argmin(fitness_scores)]
    # decoded_image = decode_genome(autoencoder, lowest_fitness_score_genome)
    # print(f"Iteration {i + 1}, Best image: {decoded_image} \n")

    return new_population
