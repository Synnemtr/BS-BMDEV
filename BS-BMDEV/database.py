import pandas as pd
import random

# Function to load the image vectores from the .csv-file
def load_image_vectors(file_path):
    # Skip the first row, set the header to the second row
    df = pd.read_csv(file_path, skiprows=2, sep=';', header=None)
    #df['image_name'] = df[0]
    values = df.iloc[:, 1:].values.tolist()
    return values

if __name__ == "__main__":
    file_path = "./list_attr_celeba.csv"
    image_database = load_image_vectors(file_path)
    random_row = random.choice(image_database)
    print(random_row)