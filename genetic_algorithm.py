"""
    A module implementing a genetic algorithm for optimization problems.

    This module provides functions to perform genetic algorithm operations such as
    selection, mutation, crossover, and generating new populations.
"""


import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.metrics import structural_similarity as ssim

def plot_fitness_scores(fitness_scores):
    """
    This function plots the fitness scores over generations.

    Parameters:
        fitness_scores (list): A list of fitness scores. Each score corresponds to a generation.

    Returns:
        None
    """
    generations = range(len(fitness_scores))
    plt.plot(generations, fitness_scores)
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness Score')
    plt.title('Fitness Scores Over Generations')
    plt.show()

def select_parents_low(population, fitness_scores):
    """
    This function selects the best half of the population based on the lowest fitness scores.

    Parameters:
        population (list): The current population of genomes.
        fitness_scores (list): The fitness scores of the current population.

    Returns:
        list: The selected half of the population with the lowest fitness scores.
    """
    selected_indices = np.argsort(fitness_scores)[:len(population) // 2]
    selected_population = [population[i] for i in selected_indices]
    return selected_population

def select_parents_high(population, fitness_scores):
    """
    This function selects the best half of the population based on the highest fitness scores.

    Parameters:
        population (list): The current population of genomes.
        fitness_scores (list): The fitness scores of the current population.

    Returns:
        list: The selected half of the population with the highest fitness scores.
    """
    selected_indices = np.argsort(-fitness_scores)[:len(population) // 2]
    selected_population = [population[i] for i in selected_indices]
    return selected_population

def roulette_wheel_selection(population, fitness_scores):
    """
    This function selects the best half of the population based on roulette wheel selection.

    Parameters:
        population (list): The current population of genomes.
        fitness_scores (list): The fitness scores of the current population.

    Returns:
        list: The selected half of the population based on roulette wheel selection.
    """
    while len(fitness_scores) < len(population):
        fitness_scores = np.concatenate((fitness_scores, fitness_scores))

    fitness_scores = fitness_scores[:len(population)]

    total_fitness = np.sum(fitness_scores)
    selection_probs = fitness_scores / total_fitness
    selected_indices = np.random.choice(np.arange(len(population)), size=len(population)//2, p=selection_probs)
    selected_population = [population[i] for i in selected_indices]
    return selected_population

def mutate_genome(genome, mutation_rate):
    """
    This function mutates a genome by adding a random normal value to it.

    Parameters:
        genome (list): The genome to be mutated.
        mutation_rate (float): The rate at which the genome should be mutated.

    Returns:
        list: The mutated genome.
    """
    mutated_genome = genome + np.random.normal(0, mutation_rate, genome.shape)
    return mutated_genome

def bit_flip_mutation(genome, mutation_rate):
    """
    This function mutates a genome by flipping its bits.

    Parameters:
        genome (list): The genome to be mutated.
        mutation_rate (float): The rate at which the genome should be mutated.

    Returns:
        list: The mutated genome.
    """
    genome_binary = np.unpackbits(genome.astype('uint8'))
    flip_indices = np.random.random(genome_binary.shape) < mutation_rate
    genome_binary[flip_indices] = 1 - genome_binary[flip_indices]
    mutated_genome = np.packbits(genome_binary).astype(genome.dtype)

    return mutated_genome

def single_point_crossover(parent1, parent2):
    """
    This function creates a new offspring by performing a single point crossover between two parents.

    Parameters:
        parent1 (list): The first parent genome.
        parent2 (list): The second parent genome.

    Returns:
        list: The offspring genome.
    """
    parent1 = np.array(parent1)
    parent2 = np.array(parent2)

    crossover_point = random.randint(0, len(parent1) - 1)
    child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    return child

def two_point_crossover(parent1, parent2):
    """
    This function creates a new offspring by performing a two-point crossover between two parents.

    Parameters:
        parent1 (list): The first parent genome.
        parent2 (list): The second parent genome.

    Returns:
        list: The offspring genome.
    """
    parent1 = np.array(parent1)
    parent2 = np.array(parent2)

    crossover_point1 = random.randint(0, len(parent1) - 1)
    crossover_point2 = random.randint(crossover_point1, len(parent1))

    child = np.concatenate([parent1[:crossover_point1], parent2[crossover_point1:crossover_point2], parent1[crossover_point2:]])
    return child

def uniform_crossover(parent1, parent2):
    """
    This function creates a new offspring by performing a uniform crossover between two parents.

    Parameters:
        parent1 (list): The first parent genome.
        parent2 (list): The second parent genome.

    Returns:
        list: The offspring genome.
    """
    parent1 = np.array(parent1)
    parent2 = np.array(parent2)

    child = np.empty_like(parent1)
    for i in range(len(parent1)):
        child[i] = parent1[i] if random.random() < 0.5 else parent2[i]
    return child

def generate_new_population_psnr(parents, population_size, mutation_rate):
    """
    This function generates a new population from the parents using two-point crossover and bit flip mutation.

    Parameters:
        parents (list): The parent genomes.
        population_size (int): The size of the population to be generated.
        mutation_rate (float): The rate at which the genomes should be mutated.

    Returns:
        list: The new population.
    """
    new_population = parents.copy()

    while len(new_population) < population_size:
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        child = two_point_crossover(parent1, parent2)

        if random.random() < mutation_rate:
            child = bit_flip_mutation(child, mutation_rate)

        new_population.append(child)

    return new_population

def generate_new_population_ssim(parents, population_size, mutation_rate):
    """
    This function generates a new population from the parents using uniform crossover and bit flip mutation.

    Parameters:
        parents (list): The parent genomes.
        population_size (int): The size of the population to be generated.
        mutation_rate (float): The rate at which the genomes should be mutated.

    Returns:
        list: The new population.
    """
    new_population = parents.copy()

    while len(new_population) < population_size:
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        child = uniform_crossover(parent1, parent2)

        if random.random() < mutation_rate:
            child = bit_flip_mutation(child, mutation_rate)

        new_population.append(child)

    return new_population

def generate_new_population_with_elitism(parents, population_size, mutation_rate, elitism_size, fitness_scores):
    """
    This function generates a new population from the parents using single point crossover and normal mutation, and includes the best individuals from the previous generation.

    Parameters:
        parents (list): The parent genomes.
        population_size (int): The size of the population to be generated.
        mutation_rate (float): The rate at which the genomes should be mutated.
        elitism_size (int): The number of best individuals from the previous generation to include in the new population.
        fitness_scores (list): The fitness scores of the current population.

    Returns:
        list: The new population.
    """
    parents_with_fitness = list(zip(parents, fitness_scores))
    parents_with_fitness.sort(key=lambda x: x[1], reverse=True)
    sorted_parents = [parent for parent, fitness in parents_with_fitness]
    new_population = sorted_parents[:elitism_size]

    while len(new_population) < population_size:
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        child = single_point_crossover(parent1, parent2)

        if random.random() < mutation_rate:
            child = mutate_genome(child, mutation_rate)

        new_population.append(child)

    return new_population

def mse_fitness_evaluation(population, encoded_target_images):
    """
    This function evaluates the fitness of the images in the population using mean squared error (MSE).
    A lower MSE value indicates a better fit

    Parameters:
        population (list): The current population of genomes.
        encoded_target_images (list): The target images to compare the population to.

    Returns:
        list: The fitness scores of the current population.
    """
    fitness_values = []
    for genome in encoded_target_images:
        fitnesses = [mse_fitness(genome, original_image) for original_image in population]
        fitness = max(fitnesses)
        fitness_values.append(fitness)
    return np.array(fitness_values)

def mse_fitness(decoded_target_image, original_image):
    """
    This function calculates the fitness of the decoded target image compared to the original image using Mean Squared Error (MSE).

    Parameters:
        decoded_target_image (array): The decoded target image.
        original_image (array): The original image.

    Returns:
        float: The average fitness value of the decoded target image.
    """
    fitness_values = []
    for target_vector, original_vector in zip(decoded_target_image, original_image):
        target_flat = target_vector.flatten()
        original_flat = original_vector.flatten()
        mean = np.mean((original_flat - target_flat)**2)
        fitness = 1 / (1 + mean)
        fitness_values.append(fitness)
    return np.mean(fitness_values)

def genetic_algorithm_with_mse(population, victim_choice, population_size, mutation_rate):
    """
    This function runs a genetic algorithm with Mean Squared Error (MSE) as the fitness function.
    Parent selection: Selects the best genomes based on the lowest fitness score
    Crossover: Single point crossover
    Mutation: Normal distribution
    New population generation: with elitism

    Parameters:
        population (list): The current population of genomes.
        victim_choice (array): The victim choice.
        population_size (int): The size of the population.
        mutation_rate (float): The rate at which the genomes should be mutated.

    Returns:
        list: The new population.
        float: The average fitness score of the new population.
    """
    elitism_size = int(0.1 * population_size) # 10% of population size

    fitness_scores = mse_fitness_evaluation(population, victim_choice)
    parents = select_parents_low(population, fitness_scores)
    new_population = generate_new_population_with_elitism(parents, population_size, mutation_rate, elitism_size, fitness_scores)

    # lowest_fitness_score_genome = population[np.argmin(fitness_scores)]
    # decoded_image = decode_genome(autoencoder, lowest_fitness_score_genome)
    # print(f"Iteration {i + 1}, Best image: {decoded_image} \n")

    average_fitness_score = np.mean(fitness_scores)
    return new_population, average_fitness_score

def psnr_fitness_evaluation(population, encoded_target_images):
    """
    This function evaluates the fitness of the images in the population using Peak Signal-to-Noise Ratio (PSNR).
    A higher PSNR value indicates a better fit.

    Parameters:
        population (list): The current population of genomes.
        encoded_target_images (list): The target images to compare the population to.

    Returns:
        array: The fitness scores of the current population.
    """
    fitness_values = []
    for genome in encoded_target_images:
        fitnesses = [psnr_fitness(genome, original_image) for original_image in population]
        fitness = max(fitnesses)
        fitness_values.append(fitness)
    return np.array(fitness_values)

def psnr_fitness(decoded_target_image, original_image):
    """
    This function calculates the fitness of the decoded target image compared to the original image using Peak Signal-to-Noise Ratio (PSNR).

    Parameters:
        decoded_target_image (array): The decoded target image.
        original_image (array): The original image.

    Returns:
        float: The average fitness value of the decoded target image.
    """
    fitness_values = []
    for target_vector, original_vector in zip(decoded_target_image, original_image):
        target_flat = target_vector.flatten()
        original_flat = original_vector.flatten()
        mse = np.mean((original_flat - target_flat)**2)
        if mse == 0:
            return 100
        MAX_I = 255.0
        psnr = 20 * np.log10(MAX_I) - 10 * np.log10(mse)
        fitness_values.append(psnr)
    return np.mean(fitness_values)

def genetic_algorithm_with_psnr(population, victim_choice, population_size, mutation_rate):
    """
    This function runs a genetic algorithm with Peak Signal-to-Noise Ratio (PSNR) as the fitness function.
    Parent selection: Selects the best genomes based on the highest fitness score
    Crossover: Two-point crossover
    Mutation: Bit flip mutation
    New population generation: without elitism

    Parameters:
        population (list): The current population of genomes.
        victim_choice (array): The victim choice.
        population_size (int): The size of the population.
        mutation_rate (float): The rate at which the genomes should be mutated.

    Returns:
        list: The new population.
        float: The average fitness score of the new population.
    """

    fitness_scores = psnr_fitness_evaluation(population, victim_choice)
    parents = select_parents_high(population, fitness_scores)
    new_population = generate_new_population_psnr(parents, population_size, mutation_rate)

    average_fitness_score = np.mean(fitness_scores)

    return new_population, average_fitness_score

def ssim_fitness_evaluation(population, encoded_target_images):
    """
    This function evaluates the fitness of the images in the population using Structural Similarity Index (SSIM).
    A higher SSIM value indicates a better fit. 

    Parameters:
        population (list): The current population of genomes.
        encoded_target_images (list): The target images to compare the population to.

    Returns:
        array: The fitness scores of the current population.
    """
    fitness_values = []
    for genome in encoded_target_images:
        fitnesses = [ssim_fitness(genome, original_image) for original_image in population]
        fitness = max(fitnesses)
        fitness_values.append(fitness)
    return np.array(fitness_values)

def ssim_fitness(decoded_target_image, original_image):
    """
    This function calculates the fitness of the decoded target image compared to the original image using Structural Similarity Index (SSIM).

    Parameters:
        decoded_target_image (array): The decoded target image.
        original_image (array): The original image.

    Returns:
        float: The average fitness value of the decoded target image.
    """
    fitness_values = []
    for target_vector, original_vector in zip(decoded_target_image, original_image):
        target_flat = target_vector.flatten()
        original_flat = original_vector.flatten()
        s = ssim(original_flat, target_flat, data_range=1.0
                 )
        fitness_values.append(s)
    return np.mean(fitness_values)

def genetic_algorithm_with_ssim(population, victim_choice, population_size, mutation_rate):
    """
    This function runs a genetic algorithm with Structural Similarity Index (SSIM) as the fitness function.
    Parent selection: Roulette wheel selection with probability based on fitness score
    Crossover: Uniform crossover
    Mutation: Bit flip mutation
    New population generation: without elitism
    
    Parameters:
        population (list): The current population of genomes.
        victim_choice (array): The victim choice.
        population_size (int): The size of the population.
        mutation_rate (float): The rate at which the genomes should be mutated.

    Returns:
        list: The new population.
        float: The average fitness score of the new population.
    """

    fitness_scores = ssim_fitness_evaluation(population, victim_choice)
    parents = roulette_wheel_selection(population, fitness_scores)
    new_population = generate_new_population_ssim(parents, population_size, mutation_rate)

    average_fitness_score = np.mean(fitness_scores)

    return new_population, average_fitness_score