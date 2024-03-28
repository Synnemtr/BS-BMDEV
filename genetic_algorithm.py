"""
    A module implementing a genetic algorithm for optimization problems.

    This module provides functions to perform genetic algorithm operations such as
    selection, mutation, crossover, and generating new populations.
"""


import numpy as np
import random

# Use autoencoder to decode the genomes
def decode_genome(decoder, genome):
    """
    Decode a genome using a specified decoder.

    Parameters:
        decoder (model): The decoder model to use for decoding.
        genome (numpy.ndarray): The genome to decode.

    Returns:
        numpy.ndarray: The decoded image.
        """
    decoded_image = decoder.predict(genome.reshape(1, 20, 18, 64))
    reshaped_image = decoded_image.reshape(160, 144, 3)
    return reshaped_image

# Evaluate the fitness of the decoded images
def fitness_evaluation(decoder, population, encoded_target_images):
    """
    Evaluate fitness of decoded images.

    Parameters:
        decoder (model): The decoder model to use for decoding.
        population (list): List of decoded images.
        encoded_target_images (list): List of target images to compare with.

    Returns:
        np.array : Array of fitness scores.
        """
    fitness_values = []
    for genome in encoded_target_images:
        # TODO: compare decoded population and target images?
        # decoded_genome = decode_genome(decoder, genome)
        fitnesses = [calculate_fitness(genome, original_image) for original_image in population]
        fitness = max(fitnesses)
        fitness_values.append(fitness)
    return np.array(fitness_values)

# Calculate the fitness of the decoded images
def calculate_fitness(decoded_target_image, original_image):
    """
    Calculate fitness of decoded images.

    Parameters:
        decoded_target_image (numpy.ndarray): Decoded target image.
        original_image (numpy.ndarray): Original image.

    Returns:
        float : Fitness value.
    """
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
    """
    Select parents based on fitness scores.

    Parameters:
        population (list) : List of individuals in the population.
        fitness_scores (numpy.ndarray) : Array of fitness scores corresponding to each individual.

    Returns:
        list : List of selected parents.
    """
    selected_indices = np.argsort(fitness_scores)[:len(population) // 2]
    selected_population = [population[i] for i in selected_indices]
    return selected_population

# Mutate genomes to create new offspring
def mutate_genome(genome, mutation_rate):
    """
    Mutate a genome with a specified mutation rate.

    Parameters:
        genome (numpy.ndarray): The genome to mutate.
        mutation_rate (float): Rate of mutation.

    Returns:
        numpy.ndarray: Mutated genome.
    """
    mutated_genome = genome + np.random.normal(0, mutation_rate, genome.shape)
    return mutated_genome

# Create offspring by crossover and mutation
def crossover(parent1, parent2):
    """
    Perform crossover between two parents to produce a child.

    Parameters:
        parent1 : First parent.
        parent2 : Second parent.

    Returns:
        numpy.ndarray : Child genome resulting from crossover.
    """
    parent1 = np.array(parent1)
    parent2 = np.array(parent2)

    crossover_point = random.randint(0, len(parent1) - 1)
    child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    return child

# Generate new population from parents
def generate_new_population(parents, population_size, mutation_rate):
    """
    Generate new population from parents.

    Parameters:
        parents (list): List of parent genomes.
        population_size (int): Desired size of the new population.
        mutation_rate (float): Rate of mutation.

    Returns:
        list : List of genomes in the new population.
    """
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
    """
    Perform genetic algorithm operations.

    Parameters:
        decoder (model): The decoder model to use for decoding.
        population (list): List of individuals in the population.
        victim_choice (list): List of target images for fitness evaluation.
        population_size (int): Desired size of the population.
        mutation_rate (float): Rate of mutation.

    Returns:
        list : New population after genetic algorithm operations.
    """
    fitness_scores = fitness_evaluation(decoder, population, victim_choice)
    parents = select_parents(population, fitness_scores)
    new_population = generate_new_population(parents, population_size, mutation_rate)

    # lowest_fitness_score_genome = population[np.argmin(fitness_scores)]
    # decoded_image = decode_genome(autoencoder, lowest_fitness_score_genome)
    # print(f"Iteration {i + 1}, Best image: {decoded_image} \n")

    return new_population
