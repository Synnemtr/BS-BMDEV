# BS-BMDEV

## How to run the project

Open Docker Desktop on your computer. In the Dockerfile in this directory, put the name of the file that should be run when using the `./run.sh` command.

Make `run.sh` executable: `chmod +x run.sh`

Run `./run.sh`
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


## The Genetic algorithm (Image Processing)
This Python code includes three different implementations of genetic algorithms for image processing. Each algorithm uses a different fitness function to evaluate the quality of the images, and different methods for parent selection, crossover, mutation, and new population generation.

Genetic Algorithm with Mean Squared Error (MSE)
This algorithm uses the Mean Squared Error (MSE) as the fitness function. MSE is a popular method to measure the error of an estimator and is calculated as the average squared difference between the estimated values and the actual value. In this context, a lower MSE value indicates a better fit.

For parent selection, this algorithm selects the best genomes based on the lowest fitness score. It uses single point crossover for mating and normal distribution for mutation. The new population is generated with elitism, meaning the best individuals from the previous generation are included in the new population.

Genetic Algorithm with Peak Signal-to-Noise Ratio (PSNR)
This algorithm uses the Peak Signal-to-Noise Ratio (PSNR) as the fitness function. PSNR is an engineering term for the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation. In this context, a higher PSNR value indicates a better fit.

For parent selection, this algorithm selects the best genomes based on the highest fitness score. It uses two-point crossover for mating and bit flip mutation for mutation. The new population is generated without elitism.

Genetic Algorithm with Structural Similarity Index (SSIM)
This algorithm uses the Structural Similarity Index (SSIM) as the fitness function. SSIM is a method for comparing similarities between two images. The SSIM index is a full reference metric; in other words, the measurement or prediction of image quality is based on an initial uncompressed or distortion-free image as reference. In this context, a higher SSIM value indicates a better fit.

For parent selection, this algorithm uses roulette wheel selection, where the probability of an individual being selected is proportional to its fitness score. It uses uniform crossover for mating and bit flip mutation for mutation. The new population is generated without elitism.

Each of these algorithms can be used depending on the specific requirements of your image processing task.

## UI

# Requirements :
Tensorflow : 2.15.0
Keras : 2.15.0
