""""======Modules======="""
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator # functional model
from keras import layers
from keras.models import Model, load_model

"""====Functions===="""
def split_data(im_fold, seed_nb):
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
        horizontal_flip=True,
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
        target_size=(218,178),
        batch_size=20,
        subset='training',
        class_mode='input',
        seed=seed_nb
    )
    val_data=val_augment.flow_from_directory(
        im_fold,
        target_size=(218,178),
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

def create_modele(shape, batch_size):
    """
    Create the layers of the model, compile it and print the resume

    Parameters :
        shape (tuples): shape of the inputs
        batch_size (int) : size of batch

    Return:
        autoencoder : the untrain model

    """
    input=layers.Input(shape=shape, batch_size=batch_size, name="encoder_input")

    # Encoder
    encoder = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
    encoder  = layers.MaxPooling2D((2, 2), padding="same")(encoder)
    encoder = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(encoder)
    encoder = layers.MaxPooling2D((2, 2), padding="same", name='encoder_output')(encoder)

    # Decoder
    decoder = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(encoder)
    decoder = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(decoder)
    decoder = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="valid", name='decoder_output')(decoder)

    # Autoencoder
    autoencoder = Model(input, decoder)
    autoencoder.compile(optimizer="adam", loss="mean_squared_error")
    autoencoder.summary()
    return autoencoder

def train_model(train_data, val_data, model, nbr_epochs, batch_size):
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
        batch_size=batch_size,
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


# """====Main===="""
# train_or_not=input("Do you want to train a new model [y/n] : ")
# print("Proceed to split data :")
# folder="./data/small_set"
# train_data, val_data=split_data(folder, seed_nb=40)
# print("Test images loaded in train data : ")
# display_data_set(train_data)
# print("Test images loaded in val data : ")
# display_data_set(val_data)
# if train_or_not=="y":
#     print("Creation of the model and print the summary : ")
#     autoencoder=create_modele((218,178,3),20)
#     history=train_model(train_data, val_data, autoencoder, 3, 20)
#     autoencoder.save("autoencoder_model.keras")
#     visualize_prediction(val_data[0][0], autoencoder, train=False)
#     plot_loss(history)
# else :
#     autoencoder_loaded, encoder, decoder=load_autoencoder_model("autoencoder_model.keras", "max_pooling2d_1",["conv2d_transpose","conv2d_2"] )
#     decoder.summary()
#     visualize_prediction(val_data[0][0], autoencoder_loaded, train=False)
#     test_encoder_decoder(val_data[0][0], encoder, decoder)