import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
from autoencoder import create_modele, load_autoencoder_model, split_data, display_data_set, visualize_prediction, test_encoder_decoder, train_model, plot_loss
from genetic_algorithm import genetic_algorithm


# Function to train the autoencoder
def train_autoencoder(train_data, val_data):
    print("Creation of the model and print the summary : ")
    autoencoder=create_modele((218,178,3),20)
    history=train_model(train_data, val_data, autoencoder, 3, 20)
    autoencoder.save("autoencoder_model.keras")
    visualize_prediction(val_data[0][0], autoencoder, train=False)
    plot_loss(history)

# Initialize population with random genomes
def population_initiation(image_folder, population_size):
    folder = image_folder + "/small_set"
    all_files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    population_files = random.sample(all_files, population_size)

    population_images = []
    plt.figure(figsize=(10, 10))
    for i, image_file in enumerate(population_files):
        img = Image.open(os.path.join(folder, image_file))
        population_images.append(np.array(img))

    return population_images

# [temporary] get the user's choice of the image that most resembles the attacker
def get_victim_choice(images):
    while True:
        print("Choose the image(s) that most resemble the attacker (or type 'quit' to exit): ")
        for i, img in enumerate(images):
            plt.subplot(2, 2, i + 1)
            img = np.squeeze(img)
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"Image {i + 1}")

        plt.show()

        choices = input("Enter the number of the image(s) that most resemble the attacker (separated by commas): ").strip()
        if choices.lower() == 'quit':
            print("Exiting...")
            return None

        choices = [int(choice.strip()) - 1 for choice in choices.split(",")]

        choice = []
        for index in choices:
            try:
                choice.append(images[index])
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
def idenfity_attacker(encoder, decoder, population_size, max_iterations, mutation_rate):
    population = population_initiation(folder, population_size)  # init random population
    
    for i in range(max_iterations):
        victim_choice = get_victim_choice(population)
        encode_victim_choice = [encoder.predict(image.reshape(1, 218, 178, 3)) for image in victim_choice] #(batch size, height, width, channels)
        encode_population = [encoder.predict(image.reshape(1, 218, 178, 3)) for image in population]
        new_population = genetic_algorithm(decoder, encode_population, encode_victim_choice, population_size, mutation_rate)
        decoded_new_population = [decoder.predict(image.reshape(1, 55, 45, 32)) for image in new_population]
        # display_image_vectors(decoded_new_population)
        population = decoded_new_population
        print(f"Iteration {i + 1} \n")


# Main function to run the program
if __name__ == "__main__":
    train_or_not=input("Do you want to train a new model [y/n] : ")
    print("Proceed to split data :")
    folder="./data/small_set"
    train_data, val_data=split_data(folder, seed_nb=40)
    print("Test images loaded in train data : ")
    display_data_set(train_data)
    print("Test images loaded in val data : ")
    display_data_set(val_data)

    if train_or_not=="y":
        train_autoencoder(train_data, val_data)
    else :
        population_size = 4
        max_iterations = 10
        mutation_rate = 0.1
        autoencoder_loaded, encoder, decoder=load_autoencoder_model("autoencoder_model.keras", "max_pooling2d_1",["conv2d_transpose","conv2d_2"] )
        encoder.summary()
        decoder.summary()
        idenfity_attacker(encoder, decoder, population_size, max_iterations, mutation_rate)
        # visualize_prediction(val_data[0][0], autoencoder_loaded, train=False)
        # test_encoder_decoder(val_data[0][0], encoder, decoder)
