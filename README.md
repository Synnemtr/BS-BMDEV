# BS-BMDEV

## How to run the project

## The autoencoder

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