# BS-BMDEV

## How to run the project

## The autoencoder

## The Genetic algorithm
This Python script employs a genetic algorithm to generate images that closely resemble a target image. The genetic algorithm, inspired by the process of natural selection, is used as an optimization technique to find the best solution to a problem.

The script starts by initializing a population of random genomes, each representing a potential solution - in this case, an image. These genomes are decoded into images using an autoencoder. The fitness of each image in the population is then evaluated based on how closely it resembles the target image. This is done by comparing the mean squared error between the target image and the generated image.

The algorithm then enters a loop where it continually selects the best genomes based on their fitness scores, uses them to generate a new population, and introduces random mutations to create variations. This process of selection, crossover, and mutation is repeated for a specified number of iterations to continually refine the solutions.

A unique aspect of this script is the inclusion of a human-in-the-loop (HITL) function, which allows a human user to interactively select the image that most closely resembles the target image. This adds a level of subjective judgment to the otherwise purely mathematical optimization process.

The script concludes by setting parameters for the genetic algorithm, such as the size of the population, the maximum number of iterations, and the mutation rate, and then running the algorithm. It also loads a database of images that are used to initialize the population of genomes.

For the moment the code uses a dummy autoencoder and data read from the list_attr_celeba.csv-file. 

## UI