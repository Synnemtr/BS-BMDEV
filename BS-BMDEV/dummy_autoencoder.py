import numpy as np

class DummyAutoencoder:
    def __init__(self, latent_dimension):
        self.latent_dimension = latent_dimension

    def decode(self, genome):
        # Ensure that the decoded genome is a NumPy array that can be flattened
        decoded_genome = np.array(genome)
        return decoded_genome