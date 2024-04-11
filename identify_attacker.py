import numpy as np
import matplotlib.pyplot as plt
import random
from autoencoder import create_autoencoder, load_autoencoder_model, split_data, display_data_set, visualize_prediction, test_encoder_decoder, train_model
from genetic_algorithm import genetic_algorithm_with_mse, genetic_algorithm_with_psnr, genetic_algorithm_with_ssim, plot_fitness_scores
import tkinter as tk
from ui import UserInterface


chosen_images_history = [] # List to store the images chosen by the user

def train_autoencoder(train_data, val_data, image_width, image_height):
    train_new =input("Do you want to train a new model [y/n] : ")
    if train_new=="y":
        saving_name=input("Choose a name for the model : ")
        print("Creation of the model and print the summary : ")
        autoencoder=create_autoencoder((image_width,image_height,3), latent_dim=256)
        train_model(train_data, val_data, autoencoder, 15, 300, saving_name)
        visualize_prediction(val_data[0][0], autoencoder, train=False, nbr_images_displayed=8)
    else :
        file_name = input("Enter the model file name : ")
        autoencoder_loaded, encoder, decoder=load_autoencoder_model('model/' + file_name + '.keras')
        train_model(train_data, val_data, autoencoder_loaded, 10, 500 , saving_name=file_name)
        visualize_prediction(val_data[0][0], autoencoder_loaded, train=False, nbr_images_displayed=8)

# Initialize population with random genomes
def population_initiation(batch, population_size):
    images, _ = next(batch)
    if population_size > len(images):
        print(f"Population size is greater than the number of images in the batch. Displaying {len(images)} images instead.")
        population_size = len(images)
    init_population = random.sample(list(images), population_size)
    return init_population

def identifying_loop(root, ui, encoder, decoder, population, population_size, max_iterations, mutation_rate):
    for i in range(max_iterations):
        # Checks the status of the choices_validated variable and if the window is still open
        while ui.window_exists and not ui.choices_validated:
            root.update_idletasks()
            root.update()
        if not ui.window_exists:
            break
        victim_choice = [ui.population[image_number - 1] for image_number in ui.user_choice]
        encode_victim_choice = [encoder.predict(image.reshape(1, 128, 128, 3)) for image in victim_choice] #(batch size, height, width, channels)
        encode_population = [np.asarray(encoder.predict(image.reshape(1, 128, 128, 3))) for image in population]
        if i < max_iterations - 1:
            new_population = genetic_algorithm(decoder, encode_population, encode_victim_choice, population_size, mutation_rate)
            decoded_new_population = [decoder.predict(image[-1]) for image in new_population]
            population = [image.reshape(128, 128, 3) for image in decoded_new_population]
            ui.display_new_images(population)    
        else:
            new_population = genetic_algorithm(decoder, encode_population, encode_victim_choice, 1, mutation_rate)
            decoded_new_population = [decoder.predict(image[-1]) for image in new_population]
            population = [image.reshape(128, 128, 3) for image in decoded_new_population]
            # new_population_if_continue = genetic_algorithm(decoder, encode_population, encode_victim_choice, population_size, mutation_rate)
            # decoded_new_population_if_continue = [decoder.predict(image[-1]) for image in new_population_if_continue]
            # population_if_continue = [image.reshape(160, 144, 3) for image in decoded_new_population_if_continue]

        ui.end_screen(population) 

# Identify the attacker using genetic algorithm and the autoencoder's encoder and decoder layer's
def idenfity_attacker(autoencoder, encoder, decoder, batch, population_size, max_iterations, mutation_rate):
    decoded_population = [autoencoder.predict(individual.reshape(1, 128, 128, 3)) for individual in population_initiation(batch, population_size)]  # init random population
    population = [image.reshape(128, 128, 3) for image in decoded_population]
    root = tk.Tk()
    ui = UserInterface(root, population)
    identifying_loop(root, ui, encoder, decoder, population, population_size, max_iterations, mutation_rate)
    # while ui.window_exists and not ui.more_iterations:
    #     root.update_idletasks()
    #     root.update()
    # if ui.more_iterations:
    #     identifying_loop(root, ui, encoder, decoder, population, population_size, max_iterations, mutation_rate)

# [temporary] get the user's choice of the image that most resembles the attacker
def get_victim_choice(mutated_images, extra_images):
    global chosen_images_history
    combined_images = mutated_images + extra_images
    while True:
        choices = input("Enter the number of the image(s) that most resemble the attacker (separated by commas): ").strip()
        if choices.lower() == 'quit':
            print("Exiting...")
            return None
        choice = []
        for index in choices:
            try:
                chosen_image = combined_images[index]
                choice.append(chosen_image)
                chosen_images_history.append(chosen_image)
            except IndexError:
                print("Invalid image number. Please try again.")
                continue

        return choice

# Display image vectors
def display_image_vectors(images):
    for i, img in enumerate(images):
        img = np.squeeze(img)
        print(f"Image {i + 1}: ")
        print(img)

# Identify the attacker using genetic algorithm and the autoencoder's encoder and decoder layer's
def idenfity_attacker(autoencoder, encoder, decoder, image_width, image_height, image_channels, batch, init_size, population_size, extra_generating, max_iterations, mutation_rate):
    population = population_initiation(batch, init_size)  # init random population
    average_fitness_scores_over_generations = []

    # User's choice of genetic algorithm
    print("Choose a genetic algorithm:")
    print("1. MSE")
    print("2. PSNR")
    print("3. SSIM")
    choice = int(input("Enter your choice (1, 2, or 3): "))

    for i in range(max_iterations):
        extra_features = population_initiation(batch, extra_generating)

        victim_choice = get_victim_choice(population, extra_features)
        encode_victim_choice = [encoder.predict(image.reshape(1, image_width, image_height, image_channels)) for image in victim_choice] #(batch size, height, width, channels)
        encode_population = [encoder.predict(image.reshape(1, image_width, image_height, image_channels)) for image in population]
        
        # Use chosen genetic algorithm
        if choice == 1:
            new_population, average_fitness_score = genetic_algorithm_with_mse(encode_population, encode_victim_choice, population_size, mutation_rate)
        elif choice == 2:
            new_population, average_fitness_score = genetic_algorithm_with_psnr(encode_population, encode_victim_choice, population_size, mutation_rate)
        elif choice == 3:
            new_population, average_fitness_score = genetic_algorithm_with_ssim(encode_population, encode_victim_choice, population_size, mutation_rate)
        else:
            print("Invalid choice. Defaulting to MSE.")
            new_population, average_fitness_score = genetic_algorithm_with_mse(encode_population, encode_victim_choice, population_size, mutation_rate)
        
        decoded_new_population = [decoder.predict(image[-1]) for image in new_population]
        average_fitness_scores_over_generations.append(average_fitness_score)

        # display_image_vectors(decoded_new_population)
        reshaped_population = [np.reshape(img, (image_width, image_height, image_channels)) for img in decoded_new_population]
        plt.figure(figsize=(10, 10))
        for i, img in enumerate(reshaped_population):
            plt.subplot(int(np.sqrt(len(reshaped_population))), int(np.sqrt(len(reshaped_population))), i + 1)
            plt.imshow(img)
            plt.axis("off")
        plt.show()

        population = decoded_new_population
    
    # Plot the fitness scores over the generations
    plot_fitness_scores(average_fitness_scores_over_generations)
            

# Main function to run the program
if __name__ == "__main__":

    # Static parameters
    population_size = 4 # number of images to display
    init_size = 9 # number of images to display in init population
    extra_images_generated = 4 # number of extra images to generate
    max_iterations = 3 # number of iterations to run the genetic algorithm

    mutation_rate = 0.01
    image_width = 128
    image_height = 128
    image_channels = 3

    print("Proceed to split data :")
    folder="./data/img_align_celeba"
    train_data, val_data=split_data(folder, seed_nb=40, image_size=(image_width,image_height), batch_size=128)

    # print("Test images loaded in train data : ")
    # display_data_set(train_data, population_size)
    # print("Test images loaded in val data : ")
    # display_data_set(val_data, population_size)
    train_or_not=input("Do you want to train a model [y/n] : ")

    if train_or_not=="y":
        train_autoencoder(train_data, val_data, image_width, image_height)
    else :
        file_name = input("Enter the model file name : ")
        autoencoder_loaded, encoder, decoder=load_autoencoder_model('model/' + file_name + '.keras')
        decoder.summary()
        autoencoder_loaded.summary()

        idenfity_attacker(autoencoder_loaded, encoder, decoder, train_data, population_size, max_iterations, mutation_rate)
        idenfity_attacker(autoencoder_loaded, encoder, decoder, image_width, image_height, image_channels, train_data, init_size, population_size, extra_images_generated, max_iterations, mutation_rate)
        # visualize_prediction(val_data[0][0], autoencoder_loaded, train=False, nbr_images_displayed=8)
        # test_encoder_decoder(val_data[0][0], encoder, decoder, 8)
