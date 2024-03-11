# BS-BMDEV

## How to run the project

## Identification software
identify_attacker.py is a program that uses a combination of an autoencoder and a genetic algorithm to identify an "attacker" image from a set of images.

The autoencoder is trained based on user input, with the option to train a new model or use an existing one. The model is then trained and its predictions are visualized.

A population of images is initialized randomly from a batch. The user is then asked to select the image(s) that most resemble the attacker.

The genetic algorithm and the autoencoder's encoder and decoder layers are used to identify the attacker. The algorithm initializes a random population, enters a loop for a maximum number of iterations, gets the victim's choice, encodes the victim's choice and the population, applies the genetic algorithm to generate a new population, decodes the new population, displays the new population, and updates the population.

The main function of the script sets the parameters for the genetic algorithm, splits the data into a training set and a validation set, asks the user whether they want to train a model, and either trains a new model or loads an existing model and identifies the attacker.

## The autoencoder

## The Genetic algorithm
This Python script employs a genetic algorithm to generate images that closely resemble a target image. The genetic algorithm, inspired by the process of natural selection, is used as an optimization technique to find the best solution to a problem.

The script starts by initializing a population of random genomes, each representing a potential solution - in this case, an image. These genomes are decoded into images using an autoencoder. The fitness of each image in the population is then evaluated based on how closely it resembles the target image. This is done by comparing the mean squared error between the target image and the generated image.

The algorithm then enters a loop where it continually selects the best genomes based on their fitness scores, uses them to generate a new population, and introduces random mutations to create variations. This process of selection, crossover, and mutation is repeated for a specified number of iterations to continually refine the solutions.

A unique aspect of this script is the inclusion of a human-in-the-loop (HITL) function, which allows a human user to interactively select the image that most closely resembles the target image. This adds a level of subjective judgment to the otherwise purely mathematical optimization process.

The script concludes by setting parameters for the genetic algorithm, such as the size of the population, the maximum number of iterations, and the mutation rate, and then running the algorithm. It also loads a database of images that are used to initialize the population of genomes.

## UI