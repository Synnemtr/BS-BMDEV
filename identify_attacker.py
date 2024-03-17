import numpy as np
import matplotlib.pyplot as plt
import random
from autoencoder import create_autoencoder, load_autoencoder_model, split_data, display_data_set, visualize_prediction, test_encoder_decoder, train_model
from genetic_algorithm import genetic_algorithm
import tkinter as tk
from ui import UserInterface


# Function to train the autoencoder
# def population_initiation(image_folder, population_size):
    # folder = image_folder + "/small_set"
    # all_files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    # population_files = random.sample(all_files, population_size)

    # population_images = []
    # plt.figure(figsize=(10, 10))
    # for i, image_file in enumerate(population_files):
    #     img = Image.open(os.path.join(folder, image_file))
    #     population_images.append(np.array(img))
    #     plt.subplot(2, 2, i + 1)
    #     img = np.squeeze(img)
    #     plt.imshow(img)
    #     plt.axis("off")
    #     plt.title(f"Image {i + 1}")

    # plt.show()

    # return population_images

def train_autoencoder(train_data, val_data):
    train_new =input("Do you want to train a new model [y/n] : ")
    if train_new=="y":
        saving_name=input("Choose a name for the model : ")
        print("Creation of the model and print the summary : ")
        autoencoder=create_autoencoder((160,144,3), latent_dim=252)
        train_model(train_data, val_data, autoencoder, 10, 500, saving_name)
        # visualize_prediction(val_data[0][0], autoencoder, train=False, nbr_images_displayed=8)
    else :
        file_name = input("Enter the model file name : ")
        autoencoder_loaded, encoder, decoder=load_autoencoder_model('model/' + file_name + '.keras')
        train_model(train_data, val_data, autoencoder_loaded, 3, 1000, saving_name=file_name)
        # visualize_prediction(val_data[0][0], autoencoder_loaded, train=False, nbr_images_displayed=8)

# Initialize population with random genomes
def population_initiation(batch, population_size):
    images, _ = next(batch)
    if population_size > len(images):
        print(f"Population size is greater than the number of images in the batch. Displaying {len(images)} images instead.")
        population_size = len(images)
    init_population = random.sample(list(images), population_size)
    return init_population

# Display image vectors
def display_image_vectors(images):
    for i, img in enumerate(images):
        img = np.squeeze(img)
        print(f"Image {i + 1}: ")
        print(img)

# Identify the attacker using genetic algorithm and the autoencoder's encoder and decoder layer's
def idenfity_attacker(autoencoder, encoder, decoder, batch, population_size, max_iterations, mutation_rate):
    population = population_initiation(batch, population_size)  # init random population
    root = tk.Tk()
    ui = UserInterface(root, population)
    # running = True
    # while running :
    for i in range(max_iterations):
        # # Check if the window is still open
        # if not root.winfo_exists():
        #     break
        while not ui.choices_validated:
            root.update_idletasks()
            root.update()
        victim_choice = [ui.population[image_number - 1] for image_number in ui.user_choice]
        # print(f"Victim choice: {victim_choice}")
        # print("Victim choice[0] type : ", type(victim_choice[0]))
        # print("Victim choice[0] shape : ", victim_choice[0].shape)
        encode_victim_choice = [np.asarray(encoder.predict(image.reshape(1, 160, 144, 3))) for image in victim_choice] #(batch size, height, width, channels)
        # print("Victim choice encoded!")
        # print("Encoded victim choice[0] type : ", type(encode_victim_choice[0]))
        # print("Encoded victim choice[0] shape : ", encode_victim_choice[0].shape)
        encode_population = [np.asarray(encoder.predict(image.reshape(1, 160, 144, 3))) for image in population]
        # print("Population encoded!")
        # print("Encoded population[0] type : ", type(encode_population[0]))
        # print("Encoded population[0] shape : ", encode_population[0].shape)
        new_population = genetic_algorithm(decoder, encode_population, encode_victim_choice, population_size, mutation_rate)
        # print("New population generated!")
        # print("New population[0] type : ", type(new_population[0]))
        # print("New population[0] shape : ", new_population[0].shape)
        # print("New population sample : ", new_population[0])
        decoded_new_population = [decoder.predict(image[-1]) for image in new_population]
        # print("New population decoded!")
        # print("Decoded population[0] type : ", type(decoded_new_population[0]))
        # print("Decoded population[0] shape : ", decoded_new_population[0].shape)
        # print("Decoded new population sample : ", decoded_new_population[0])
        population = [image.reshape(160, 144, 3) for image in decoded_new_population]
        # print("Population updated!")
        # print("Population[0] type : ", type(population[0]))
        # print("Population[0] shape : ", population[0].shape)

        ui.display_new_images(population)
            

# Main function to run the program
if __name__ == "__main__":

    population_size = 4
    max_iterations = 10
    mutation_rate = 0.1

    print("Proceed to split data :")
    folder="./data/img_align_celeba"
    train_data, val_data=split_data(folder, seed_nb=40, image_size=(160,144))

    # print("Test images loaded in train data : ")
    # display_data_set(train_data, population_size)
    # print("Test images loaded in val data : ")
    # display_data_set(val_data, population_size)
    train_or_not=input("Do you want to train a model [y/n] : ")

    if train_or_not=="y":
        train_autoencoder(train_data, val_data)
    else :
        file_name = input("Enter the model file name : ")
        autoencoder_loaded, encoder, decoder=load_autoencoder_model('model/' + file_name + '.keras')
        encoder.summary()
        decoder.summary()
        autoencoder_loaded.summary()
        idenfity_attacker(autoencoder_loaded, encoder, decoder, train_data, population_size, max_iterations, mutation_rate)
        
        # visualize_prediction(val_data[0][0], autoencoder_loaded, train=False, nbr_images_displayed=8)
        # test_encoder_decoder(val_data[0][0], encoder, decoder, 8)
