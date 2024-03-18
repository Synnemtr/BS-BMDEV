# BS-BMDEV

## How to run the project

## Identification software
identify_attacker.py is a program that uses a combination of an autoencoder and a genetic algorithm to identify an "attacker" image from a set of images.

The autoencoder is trained based on user input, with the option to train a new model or use an existing one. The model is then trained and its predictions are visualized.

A population of images is initialized randomly from a batch. The user is then asked to select the image(s) that most resemble the attacker.

The genetic algorithm and the autoencoder's encoder and decoder layers are used to identify the attacker. The algorithm initializes a random population, enters a loop for a maximum number of iterations, gets the victim's choice, encodes the victim's choice and the population, applies the genetic algorithm to generate a new population, decodes the new population, displays the new population, and updates the population.

The main function of the script sets the parameters for the genetic algorithm, splits the data into a training set and a validation set, asks the user whether they want to train a model, and either trains a new model or loads an existing model and identifies the attacker.

## The autoencoder
This Python script allows to create, train or load a variational antoencoder model for image reconstruction from celebA dataset. The autoencoder is built using Keras library with Tensorflow backend.

The first step is the loading of the dataset and splitting in a training set and a validation set. The sets are tensor object containing Numpy arrays of a number of images determine by the batch size. The first images of each set are dipslayed to check the loading of data.

Then you can choose to create a new variationnal autoencoder model. During the training the autoencoder is saved at the end of each epoch in the model file according to the name you entered and with a keras extension.

To train a previously created model, enter the name of the .keras file presents in the model file without the .keras, the model will also be saved at the end of each epoch.
At the end of the training, graph of loss value and val loss value can be displayed. Also the prediction on the validation set are displayed.

A previously trained model can be loaded without training, enter the name as previously. Predictions on the validation by the autoencoder and by the encoder followin by the decoder set will be displayed in order to check the encoder and decoder have been loaded correcly. 

## The Genetic algorithm
This Python script employs a genetic algorithm to generate images that closely resemble a target image. The genetic algorithm, inspired by the process of natural selection, is used as an optimization technique to find the best solution to a problem.

The script starts by initializing a population of random genomes, each representing a potential solution - in this case, an image. These genomes are decoded into images using an autoencoder. The fitness of each image in the population is then evaluated based on how closely it resembles the target image. This is done by comparing the mean squared error between the target image and the generated image.

The algorithm then enters a loop where it continually selects the best genomes based on their fitness scores, uses them to generate a new population, and introduces random mutations to create variations. This process of selection, crossover, and mutation is repeated for a specified number of iterations to continually refine the solutions.

A unique aspect of this script is the inclusion of a human-in-the-loop (HITL) function, which allows a human user to interactively select the image that most closely resembles the target image. This adds a level of subjective judgment to the otherwise purely mathematical optimization process.

The script concludes by setting parameters for the genetic algorithm, such as the size of the population, the maximum number of iterations, and the mutation rate, and then running the algorithm. It also loads a database of images that are used to initialize the population of genomes.

## UI
# Requirements :
Tensorflow : 2.15.0
Keras : 2.15.0



