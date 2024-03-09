""""======Modules======="""
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator 
from keras import layers
from keras.models import Model, load_model
from keras import backend as K
from keras.losses import mse
import numpy as np
import tensorflow as tf
"""====Functions===="""
def split_data(im_fold, seed_nb, image_size):
    """
    Proceed to the splitting of the data in a train set and a validation set

    Parameters :
        im_fold (string) : path to the folder containing the pictures
        seed_nb (int) : number of the seed you want to use

    Return :
        train_data (keras.src.preprocessing.image.DirectoryIterator): set of the images used to train
        val_data (keras.src.preprocessing.image.DirectoryIterator): set of the images used for validation
    """
    train_augment=ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.1,
        shear_range=0.1,
        validation_split=0.2,
    )
    val_augment=ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    train_data=train_augment.flow_from_directory(
        im_fold,
        target_size=image_size,
        batch_size=20,
        subset='training',
        class_mode='input',
        seed=seed_nb
    )
    val_data=val_augment.flow_from_directory(
        im_fold,
        target_size=image_size,
        batch_size=20,
        subset='validation',
        class_mode='input',
        seed=seed_nb
    )
    return train_data, val_data

def display_data_set(data):
    """
    Display the 8th first pictures of the 1rst batch of a set of data

    Parameters : 
        data (numpy array) : set of image data (train or validation set)
    
    """
    plt.figure(figsize=(10, 10))
    for image in data[0]:
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image[i])
            plt.axis("off")
        plt.show()
        break

@tf.function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 300))
    return z_mean + K.exp(z_log_var / 2) * epsilon

def create_encoder(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32,3,strides=2, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64,3,strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(64,3,strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(64,3,strides=2, padding='same', activation='relu')(x)

    shape_before_flattening = K.int_shape(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation = "relu")(x)

    z_mean = layers.Dense(300, name="z_mean")(x)
    z_log_var = layers.Dense(300, name="z_log_var")(x)

    z = layers.Lambda(sampling)([z_mean, z_log_var])

    encoder=Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.compile()
    encoder.summary()
    return shape_before_flattening, z, encoder

def create_decoder(shape_before_flattening, z):
    decoder_input = layers.Input(K.int_shape(z)[1:])
    x = layers.Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)
    x = layers.Reshape(shape_before_flattening[1:])(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation="relu")(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation="relu")(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation="relu")(x)
    outputs = layers.Conv2DTranspose(3, 3, padding='same', activation="sigmoid")(x)

    decoder = Model(decoder_input, outputs, name='decoder')
    decoder.compile()
    decoder.summary()
    return decoder

def create_autoencoder(input_shape):
    """
    Create the layers of the model, compile it and print the resume

    Parameters :
        input_shape (tuples): shape of the inputs
        batch_size (int) : size of batch

    Return:
        autoencoder : the untrain model

    """
    shape_before_flattening, z_layer, encoder = create_encoder(input_shape)
    decoder=create_decoder(shape_before_flattening, z_layer)

    inputs=layers.Input(shape=input_shape)
    z_mean, z_log_var, z = encoder(inputs)
    outputs=decoder(z)
    vae=Model(inputs, outputs)

    # Define the VAE loss function
    reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
    reconstruction_loss *= input_shape[0] * input_shape[1] * input_shape[2]
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)
    vae_loss = K.mean( reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.add_metric(kl_loss, name="kl_loss")
    vae.add_metric(reconstruction_loss, name="reconstruction_loss")
    vae.compile(optimizer='adam')
    vae.summary()
    return vae

def train_model(train_data, val_data, model, nbr_epochs, steps_per_epoch):
    """
    Train the model on a train set

    Parameters :
        train_data = set of data to be train
        val_data = set of data use to validate the model
        model = the compiled model
        nrb_epochs = number of epochs
        batch_size = size of the batch 
    
    Return: 
        history : history of the model training
    """
    history=model.fit(
        train_data,
        epochs=nbr_epochs,
        steps_per_epoch=steps_per_epoch,
        shuffle=True,
        validation_data=val_data,
        workers=-1 #use all the processors
    )
    return history

def visualize_prediction(data_batch, model, train):
    """
    Visualize the 8 first results of the model prediction on one batch
        
    Parameters :
        data_batch (NumPy array) = one batch of a data set
        model = the model use to predict
        train (bool) = True if its a batch from a train set /
    False if it's from a validation set

    """
    pred_batch=model.predict(data_batch)
    fig,ax=plt.subplots(4,4,figsize=(20,8))
    if train:
        plt.suptitle("Model Evaluation on Train Data", size=18)
    else:
        plt.suptitle("Model Evaluation on Validation Data", size=18)
    for i in range (8):
        plt.subplot(4,4,i*2+1)
        plt.imshow(data_batch[i])
        plt.title("Image")
        plt.axis('off')
        plt.subplot(4,4,i*2+2)
        plt.imshow(pred_batch[i])
        plt.title("Predicted")
        plt.axis('off')
    plt.show()

def load_autoencoder_model(model_path, encoder_layer, decoder_layers):
    """
    load the encoder and the decoder from an saved autoencoder

    Parameters:
        model_path (str) = path to the model file
        encoder_layer (str) = name of the last layer of the encoder
        decoder_layers (2-lists of str) = name of the first and last layers of the decoder

    Return :
        autoencoder_loaded = the full autoencoder model
        encoder = the encoder part of the autoencoder model
        decoder = the decoder part of the autoencoder model
    """
    autoencoder_loaded=load_model(model_path)
    encoder=Model(inputs=autoencoder_loaded.inputs, outputs=autoencoder_loaded.get_layer(encoder_layer).output)
    decoder=Model(inputs=autoencoder_loaded.get_layer(decoder_layers[0]).input, outputs=autoencoder_loaded.get_layer(decoder_layers[1]).output)
    return autoencoder_loaded, encoder, decoder

def test_encoder_decoder(data_batch, encoder, decoder):
    """
    Visualize the results of predictions using encoder and decoder separatly

    Parameters :
        data_batch (Numpy Array) = one batch of a data set
        encoder = the encoder part of the model
        decoder =the decoder part of the model
    """
    encoded_data=encoder.predict(data_batch)
    decoded_data=decoder.predict(encoded_data)
    for i in range (8):
        plt.subplot(4,4 ,i*2+1)
        plt.imshow(data_batch[i])
        plt.title("Image")
        plt.axis('off')
        plt.subplot(4,4,i*2+2)
        plt.imshow(decoded_data[i])
        plt.title("Predicted")
        plt.axis("off")
    plt.show()

def plot_loss(history):
    """
    Plot the loss of the model

    Parameters:
        history = history of the model training
    """
    plt.plot(history.history["loss"], label='train')
    plt.plot(history.history['val_loss'], label='Validatiion')
    plt.title('Loss of the model')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.show()


"""====Main===="""
if __name__ == "__main__":
    train_or_not=input("Do you want to train a new model [y/n] : ")
    print("Proceed to split data :")
    folder="./data/img_align_celeba"
    train_data, val_data=split_data(folder, seed_nb=40, image_size=(128,128))
    print("Test images loaded in train data : ")
    display_data_set(train_data)
    print("Test images loaded in val data : ")
    display_data_set(val_data)
    if train_or_not=="y":
        print("Creation of the model and print the summary : ")
        autoencoder=create_autoencoder((128,128,3))
        history=train_model(train_data, val_data, autoencoder, 2, 800)
        autoencoder.save("vae_model.h5")
        visualize_prediction(val_data[0][0], autoencoder, train=False)
        plot_loss(history)
    else :
        autoencoder_loaded, encoder, decoder=load_autoencoder_model("autoencoder_model.keras", "max_pooling2d_1",["conv2d_transpose","conv2d_2"] )
        decoder.summary()
        visualize_prediction(val_data[0][0], autoencoder_loaded, train=False)
        test_encoder_decoder(val_data[0][0], encoder, decoder)