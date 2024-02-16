import numpy as np
import matplotlib.pyplot as plt
import random
from database import load_image_vectors
from dummy_autoencoder import DummyAutoencoder


# Initialize population with random genomes
def population_initiation(autoencoder, database, population_size):
    indices = random.sample(range(len(database)), population_size)
    population = [database[i] for i in indices]
    return [autoencoder.decode(image) for image in population]

# Use autoencoder to decode the genomes
def decode_genome(autoencoder, genome):
    decoded_image = autoencoder.decode(genome)
    return decoded_image

# Evaluate the fitness of the decoded images
def fitness_evaluation(autoencoder, population, target_images):
    fitness_values = []
    for genome in population:
        decoded_image = decode_genome(autoencoder, genome)
        fitnesses = [calculate_fitness(autoencoder, decoded_image, target_image) for target_image in target_images]
        fitness = max(fitnesses)
        fitness_values.append(fitness)
    return np.array(fitness_values)

# Calculate the fitness of the decoded images
def calculate_fitness(autoencoder, decoded_image, target_image):
    fitness_values = []
    for decoded_vector, target_vector in zip(decoded_image, target_image):
        decoded_flat = decoded_vector.flatten()
        target_flat = target_vector.flatten()
        mean = np.mean((decoded_flat - target_flat)**2)
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

# Function to choose the most resembling image from the population
def get_HITL_choice(autoencoder, population):
    decoded_images = [decode_genome(autoencoder, genome) for genome in population]
    while True:
        print("Choose the image(s) that most resemble the attacker (or type 'quit' to exit): ")
        for i, image in enumerate(decoded_images):
            print(f"{i + 1}: Image {image}")
        
        choices = input("Enter the number of the image that most resembles the attacker: ")
        if choices.lower() == 'quit':
            print("Exiting...")
            return None

        choices = choices.split(",")
        try:
            for i in range(len(choices)):
                choices[i] = int(choices[i].strip()) - 1
        except ValueError:
            print("Invalid input. Please enter the number of the image.")
            continue

        choice = []
        for index in choices:
            try:
                choice.append(population[index])
            except IndexError:
                print("Invalid image number. Please try again.")
                continue
        
        return choice

# Genetic algorithm function
def genetic_algorithm(autoencoder, database, population_size, max_iterations, mutation_rate):
    
    population = population_initiation(autoencoder, database, population_size)

    for i in range(max_iterations):
        victim_choice = get_HITL_choice(autoencoder, population)
        fitness_scores = fitness_evaluation(autoencoder, population, victim_choice)
        parents = select_parents(population, fitness_scores)
        population = generate_new_population(parents, population_size, mutation_rate)
        
        lowest_fitness_score_genome = population[np.argmin(fitness_scores)]
        decoded_image = decode_genome(autoencoder, lowest_fitness_score_genome)
        print(f"Iteration {i + 1}, Best image: {decoded_image} \n")
        # plt.imshow(decoded_image)
        # plt.show()

if __name__ == "__main__":
    latent_dimension = 10 #of the autoencoder
    autoencoder_instance = DummyAutoencoder(latent_dimension) #dummy autoencoder
    population_size = 4
    max_iterations = 10
    mutation_rate = 0.1
    file_path = "./list_attr_celeba.csv"
    image_database = load_image_vectors(file_path) #database
    genetic_algorithm(autoencoder_instance, image_database, population_size, max_iterations, mutation_rate)