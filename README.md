# BS-BMDEV

# The autoencoder :

This Python script allows to create, train or load a variational antoencoder model for image reconstruction from celebA dataset. The autoencoder is built using Keras library with Tensorflow backend.

The first step is the loading of the dataset and splitting in a training set and a validation set. The sets are tensor object containing Numpy arrays of a number of images determine by the batch size. The first images of each set are dipslayed to check the loading of data.

Then you can choose to create a new variationnal autoencoder model. During the training the autoencoder is saved at the end of each epoch in the model file according to the name you entered and with a keras extension.

To train a previously created model, enter the name of the .keras file presents in the model file without the .keras, the model will also be saved at the end of each epoch.
At the end of the training, graph of loss value and val loss value can be displayed. Also the prediction on the validation set are displayed.

A previously trained model can be loaded without training, enter the name as previously. Predictions on the validation by the autoencoder and by the encoder followin by the decoder set will be displayed in order to check the encoder and decoder have been loaded correcly. 
