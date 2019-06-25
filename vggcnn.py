# import libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import cv2


class VGG16CNN:
    pass

    # define model variables
    epoch_size = 25
    b_size = 34
    num_split = 0.2
    file_name = 'vgg16_model/CNN_25.h5'
    image_width = 100
    image_height = 100

    # prepare data
    img = pickle.load(open("vgg16_model/Images.pickle", "rb"))
    lbl = pickle.load(open("vgg16_model/Labels.pickle", "rb"))

    # Normalize data
    IMG = img/225.0

    # creating a VGG 16 Convolution Neural network
    def create_vgg16_cnn(self):

        # create a feed forward model
        forward_model = tf.keras.Sequential()

        # Input layer
        forward_model.add(tf.keras.layers.Conv2D(8, (3, 3), input_shape=(self.image_width, self.image_height, 1)))
        forward_model.add(tf.keras.layers.Activation('relu'))
        forward_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        # hidden layer's
        forward_model.add(tf.keras.layers.Conv2D(16, (3, 3)))
        forward_model.add(tf.keras.layers.Activation('relu'))
        forward_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        forward_model.add(tf.keras.layers.Conv2D(32, (3, 3)))
        forward_model.add(tf.keras.layers.BatchNormalization())
        forward_model.add(tf.keras.layers.Activation('relu'))
        forward_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        forward_model.add(tf.keras.layers.Conv2D(64, (3, 3)))
        forward_model.add(tf.keras.layers.BatchNormalization())
        forward_model.add(tf.keras.layers.Activation('relu'))
        forward_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        forward_model.add(tf.keras.layers.Flatten())
        forward_model.add(tf.keras.layers.Dense(64))
        forward_model.add(tf.keras.layers.Activation('relu'))

        forward_model.add(tf.keras.layers.Dense(1))
        forward_model.add(tf.keras.layers.Activation('sigmoid'))
        forward_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Begin training
        history = forward_model.fit(self.IMG, self.lbl, batch_size=self.b_size, epochs=self.epoch_size,
                                    validation_split=self.num_split)

        '''
        the following code plot metrics that the classier save to determine how bad and good does it perform during
        training.
        ---------------------------
        section 1: summarize history for accuracy
        section 2: # summarize history for loss
        '''
        fig = plt.figure()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        fig.savefig("vgg16_model/dataset1_accuracy25.png", dpi=fig.dpi)

        fig2 = plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        fig2.savefig('vgg16_model/dataset1_loss25.png', dpi=fig2.dpi)

        """
        -----------------------------------------
        Model History
        Confusion Matrix
        F1 score, Precision
        """

        # saving the model to a json file
        model_json = forward_model.to_json()
        with open("vgg16_model/model_25.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        forward_model.save_weights(self.file_name)
        print("Saved model to disk")


v = VGG16CNN()
v.create_vgg16_cnn()
