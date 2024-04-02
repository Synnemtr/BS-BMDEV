import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.metrics import structural_similarity as ssim


# Select the best genomes based on the lowest fitness score fitness score
def select_parents_low(population, fitness_scores):
    selected_indices = np.argsort(fitness_scores)[:len(population) // 2]
    selected_population = [population[i] for i in selected_indices]
    return selected_population

# Select the best genomes based on the highest fitness score fitness score
def select_parents_high(population, fitness_scores):
    selected_indices = np.argsort(-fitness_scores)[:len(population) // 2]
    selected_population = [population[i] for i in selected_indices]
    return selected_population

# Select the best genomes based on roulette wheel selection, where the probability of an individual being selected is proportional to its fitness score.
def roulette_wheel_selection(population, fitness_scores):
    while len(fitness_scores) < len(population):     # Repeat fitness_scores until it matches the length of the population
        fitness_scores = np.concatenate((fitness_scores, fitness_scores))

    # Trim fitness_scores to the length of the population
    fitness_scores = fitness_scores[:len(population)]

    total_fitness = np.sum(fitness_scores)
    selection_probs = fitness_scores / total_fitness
    selected_indices = np.random.choice(np.arange(len(population)), size=len(population)//2, p=selection_probs)
    selected_population = [population[i] for i in selected_indices]
    return selected_population

# Mutate genomes to create new offspring
def mutate_genome(genome, mutation_rate):
    mutated_genome = genome + np.random.normal(0, mutation_rate, genome.shape)
    return mutated_genome

# Mutate genomes with bit flip mutation - flips bits in the genome with a certain probability
def bit_flip_mutation(genome, mutation_rate):
    genome_binary = np.unpackbits(genome.astype('uint8'))
    flip_indices = np.random.random(genome_binary.shape) < mutation_rate
    genome_binary[flip_indices] = 1 - genome_binary[flip_indices]
    mutated_genome = np.packbits(genome_binary).astype(genome.dtype)

    return mutated_genome

# Create offspring by sigle point crossover
def single_point_crossover(parent1, parent2):
    parent1 = np.array(parent1)
    parent2 = np.array(parent2)

    crossover_point = random.randint(0, len(parent1) - 1)
    child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    return child

# Create offspring by two-point crossover
def two_point_crossover(parent1, parent2):
    parent1 = np.array(parent1)
    parent2 = np.array(parent2)

    crossover_point1 = random.randint(0, len(parent1) - 1)
    crossover_point2 = random.randint(crossover_point1, len(parent1))

    child = np.concatenate([parent1[:crossover_point1], parent2[crossover_point1:crossover_point2], parent1[crossover_point2:]])
    return child

# Create offspring by uniform crossover
def uniform_crossover(parent1, parent2):
    parent1 = np.array(parent1)
    parent2 = np.array(parent2)

    child = np.empty_like(parent1)
    for i in range(len(parent1)):
        child[i] = parent1[i] if random.random() < 0.5 else parent2[i]
    return child

# Generate new population from parents for psnr
def generate_new_population_psnr(parents, population_size, mutation_rate):
    new_population = parents.copy()

    while len(new_population) < population_size:
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        child = two_point_crossover(parent1, parent2)

        if random.random() < mutation_rate:
            child = bit_flip_mutation(child, mutation_rate)

        new_population.append(child)

    return new_population

# Generate new population from parents for ssim
def generate_new_population_ssim(parents, population_size, mutation_rate):
    new_population = parents.copy()

    while len(new_population) < population_size:
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        child = uniform_crossover(parent1, parent2)

        if random.random() < mutation_rate:
            child = bit_flip_mutation(child, mutation_rate)

        new_population.append(child)

    return new_population

# Generate new population with elitism including the best individuals from the previous generation
def generate_new_population_with_elitism(parents, population_size, mutation_rate, elitism_size, fitness_scores):
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


# Evaluate the fitness of the images with mean squared error (MSE)
# A lower MSE value indicates a better fit
def mse_fitness_evaluation(population, encoded_target_images):
    fitness_values = []
    for genome in encoded_target_images:
        fitnesses = [mse_fitness(genome, original_image) for original_image in population]
        fitness = max(fitnesses)
        fitness_values.append(fitness)
    return np.array(fitness_values)

def mse_fitness(decoded_target_image, original_image):
    fitness_values = []
    for target_vector, original_vector in zip(decoded_target_image, original_image):
        target_flat = target_vector.flatten()
        original_flat = original_vector.flatten()
        mean = np.mean((original_flat - target_flat)**2)
        fitness = 1 / (1 + mean)
        fitness_values.append(fitness)
    return np.mean(fitness_values)

# Genetic algorithm function
# Fitness function: Mean Squared Error (MSE)
# Parent selection: Selects the best genomes based on the lowest fitness score
# Crossover: Single point crossover
# Mutation: Normal distribution
# New population generation: with elitism
def genetic_algorithm_with_mse(population, victim_choice, population_size, mutation_rate):
    elitism_size = int(0.1 * population_size) # 10% of population size

    fitness_scores = mse_fitness_evaluation(population, victim_choice)
    parents = select_parents_low(population, fitness_scores)
    new_population = generate_new_population_with_elitism(parents, population_size, mutation_rate, elitism_size, fitness_scores)

    # lowest_fitness_score_genome = population[np.argmin(fitness_scores)]
    # decoded_image = decode_genome(autoencoder, lowest_fitness_score_genome)
    # print(f"Iteration {i + 1}, Best image: {decoded_image} \n")

    return new_population, fitness_scores


# Evaluate the fitness of the images with Peak Signal-to-Noise Ratio (PSNR)
# A higher PSNR value indicates a better fit
def psnr_fitness_evaluation(population, encoded_target_images):
    fitness_values = []
    for genome in encoded_target_images:
        fitnesses = [psnr_fitness(genome, original_image) for original_image in population]
        fitness = max(fitnesses)
        fitness_values.append(fitness)
    return np.array(fitness_values)

def psnr_fitness(decoded_target_image, original_image):
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

# Genetic algorithm function
# Fitness function: Peak Signal-to-Noise Ratio (PSNR)
# Parent selection: Selects the best genomes based on the highest fitness score
# Crossover: Two-point crossover
# Mutation: Bit flip mutation
# New population generation: without elitism
def genetic_algorithm_with_psnr(population, victim_choice, population_size, mutation_rate):

    fitness_scores = psnr_fitness_evaluation(population, victim_choice)
    parents = select_parents_high(population, fitness_scores)
    new_population = generate_new_population_psnr(parents, population_size, mutation_rate)

    return new_population, fitness_scores


# Evaluate the fitness of the images with Structural Similarity Index (SSIM)
# A higher SSIM value indicates a better fit
def ssim_fitness_evaluation(population, encoded_target_images):
    fitness_values = []
    for genome in encoded_target_images:
        fitnesses = [ssim_fitness(genome, original_image) for original_image in population]
        fitness = max(fitnesses)
        fitness_values.append(fitness)
    return np.array(fitness_values)

def ssim_fitness(decoded_target_image, original_image):
    fitness_values = []
    for target_vector, original_vector in zip(decoded_target_image, original_image):
        target_flat = target_vector.flatten()
        original_flat = original_vector.flatten()
        s = ssim(original_flat, target_flat, data_range=1.0
                 )
        fitness_values.append(s)
    return np.mean(fitness_values)


# Genetic algorithm function
# Fitness function: Structural Similarity Index (SSIM)
# Parent selection: Roulette wheel selection with probability based on fitness score
# Crossover: Uniform crossover
# Mutation: Bit flip mutation
# New population generation: without elitism
def genetic_algorithm_with_ssim(population, victim_choice, population_size, mutation_rate):

    fitness_scores = ssim_fitness_evaluation(population, victim_choice)
    parents = roulette_wheel_selection(population, fitness_scores)
    new_population = generate_new_population_ssim(parents, population_size, mutation_rate)

    return new_population, fitness_scores