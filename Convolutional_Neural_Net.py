# 1st Party Modules
from pickle import load
from typing import List, Tuple
from itertools import chain

# 3rd Party Modules
import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, \
    MaxPool2D, BatchNormalization, GaussianNoise

from tensorflow.keras import regularizers

from tensorflow.keras.layers.experimental.preprocessing import RandomRotation,\
    Normalization

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["figure.figsize"] = (12.8, 7.20)


class Helper:
    @staticmethod
    def conv_gscale(data: np.array, numerator: float) -> np.array:
        """ Convert to grey scale. """

        try:
            return data / numerator
            
        except:
            raise ZeroDivisionError

    
    @staticmethod
    def add_dim(data: np.array) -> np.array:
        """ Add new dimension and convert to float 32. """
        
        return data[..., tf.newaxis].astype(np.float32)


class CNN_base(tf.keras.Model):
    
    def __init__(self, feats: List[int], k_dim: Tuple[int], p_dim: Tuple[int]):
        """ Constructor of base CNN architecture. """
        
        super(CNN_base, self).__init__()
        
        self.noise = GaussianNoise(stddev=0.05)
        self.conv_layers = [
            [
                Conv2D(feat, k_dim, activation=tf.nn.relu, padding="same",
                    kernel_regularizer=regularizers.L1(l1=0.01),
                    bias_regularizer=regularizers.L1(l1=0.01), 
                    activity_regularizer=None), 
                MaxPool2D(pool_size=p_dim, padding="same")
            ]
            for _, feat in enumerate(feats)
        ]


        self.conv_layers = list(chain.from_iterable(self.conv_layers))      
        self.batch_norm = BatchNormalization(axis=1)
        
        self.flatten = Flatten(data_format="channels_last")
        self.dense_1 = Dense(128, activation=tf.nn.sigmoid)

        self.dense_2 = Dense(10)       
     

    @tf.function
    def call(self, input_data):
        """ Define computation graph of the model. """
        
        input_data = self.noise(input_data)

        for layer in self.conv_layers:
            input_data = layer(input_data)

        out_batch = self.batch_norm(input_data)
        out_flat  = self.flatten(out_batch)
        
        out_dense_1 = self.dense_1(out_flat)
        out_dense_2 = self.dense_2(out_dense_1)

        return out_dense_2


    @staticmethod
    def plot_accuracies(hist_dict: dict) -> None:
        """ Plot the accuracies over the epochs. """

        if not isinstance(hist_dict, dict):
            raise TypeError(hist_dict)

        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, 
            tight_layout=True)

        # range of epochs
        epochs = range(1, len(hist_dict["loss"])+1)

        # Plot loss w.r.t number of epochs
        axs[0].set_ylabel("Loss")
        axs[0].plot(epochs, hist_dict["loss"])
        axs[0].plot(epochs, hist_dict["val_loss"])
        axs[0].legend(["loss", "val_loss"])
        
        # Plot accuracy w.r.t number of epochs
        axs[1].set_xlabel("Number of epochs")
        axs[1].set_ylabel("Accuracy")
        axs[1].plot(epochs, hist_dict["accuracy"])
        axs[1].plot(epochs, hist_dict["val_accuracy"])
        axs[1].legend(["accuracy", "val_accuracy"])
        
        plt.xticks(ticks=epochs)
        plt.show()
        
    
    @staticmethod
    def plot_confusion_matrix(true: np.array, pred: np.array) -> None:
        """ Plot confusion matrix. """
        
        if len(pred) == 0 or len(true) == 0:
            raise Exception("Array is empty!")
        
        if not isinstance(pred, np.ndarray):
            pred = np.array(pred)

        # Calculate confusion matrix
        conf_matrix = tf.math.confusion_matrix(true, pred).numpy()

        # Plot confusion matrix
        fig, axs = plt.subplots(tight_layout=True)
        image = axs.imshow(conf_matrix)

        # True and predicted labels
        true_labels = range(0, 10)
        pred_labels = true_labels

        # Change the location of the ticks
        axs.tick_params(top=True, bottom=False, labeltop=True, 
            labelbottom=False)

        # Set axis ticks
        axs.set_yticks(np.arange(len(true_labels)))
        axs.set_xticks(np.arange(len(pred_labels)))

        # Set axis ticks labels
        axs.set_yticklabels(true_labels)
        axs.set_xticklabels(pred_labels)

        # Set x and y label
        axs.set_ylabel("True labels", fontsize=15)
        axs.set_xlabel("Predicted labels", fontsize=15)    

        fig.suptitle("Confusion Matrix MNIST", fontsize=15, ha="center")
        fig.colorbar(image)

        for i in true_labels:
            for j in pred_labels:
                text = axs.text(j, i, conf_matrix[i, j],
                    ha="center", va="center", color="w")
    
        plt.show()


class CNN_pert(CNN_base):
    

    def __init__(self):
        super(CNN_pert, self).__init__([32, 64], (5, 5), (2, 2))

        self.normalisation = Normalization(axis=-1)
        self.data_aug = RandomRotation(factor=(-0.75,0.75),fill_mode="constant")
        self.drop_out = Dropout(0.25)
        
    
    @tf.function
    def call(self, input_data):
        
        input_data = self.normalisation(input_data)
        input_data = self.data_aug(input_data)
        input_data = self.noise(input_data)

        for layer in self.conv_layers:
            input_data = layer(input_data)

        out_batch = self.batch_norm(input_data)
        out_flat  = self.flatten(out_batch)
        
        out_dense_1 = self.dense_1(out_flat)
        out_dense_1 = self.drop_out(out_dense_1)
        out_dense_2 = self.dense_2(out_dense_1)

        return out_dense_2


if __name__ == "__main__":

    # Load data
    (x_train, y_train),(x_test, y_test) = load_data(path='mnist.npz')

    data_pert = load(open("mnist_test_perturb.pickle", "rb"))
    x_test_pert = data_pert["x_perturb"]
    y_test_pert = data_pert["y_perturb"]
    
    # Normalize data
    x_test =  Helper.add_dim(Helper.conv_gscale(x_test, 255.0))
    x_train =  Helper.add_dim(Helper.conv_gscale(x_train, 255.0))
    x_test_pert =  Helper.add_dim(Helper.conv_gscale(x_test_pert, 255.0))

    # Configurations for all Models (for better comparability)
    lr_sched = ExponentialDecay(0.01, decay_steps=100000, decay_rate=0.96)
    loss_obj = SparseCategoricalCrossentropy(from_logits=True)
    optimizer = Adam(learning_rate=lr_sched) 
    early_stop = EarlyStopping(
        monitor="val_loss", min_delta=1e-5, patience=20, verbose=1)


    # Subtask A:---------------------------------------------------------------
    model = CNN_base([32, 64], (5, 5), (2, 2))
    model.compile(optimizer=optimizer, loss=loss_obj, metrics=["accuracy"])
    history = model.fit(x_train, y_train, epochs=3, batch_size=32, 
        validation_split=0.2, callbacks=early_stop)

    # Calculate prediction and evaluate
    prediction = model.predict(x_test)
    y_pred = np.argmax(prediction, axis=1)
     
    model.evaluate(x_test, y_test, batch_size=32)

    print("Before data augmentation on the perturbed test set")
    model.evaluate(x_test_pert, y_test_pert, batch_size=32)
   
    # Plot accuracies and confusion matrix of the model 
    CNN_base.plot_accuracies(history.history)
    CNN_base.plot_confusion_matrix(y_test, y_pred)
 
    # Subtask B:---------------------------------------------------------------
    model = CNN_pert()
    model.compile(optimizer=optimizer, loss=loss_obj, metrics=["accuracy"])

    history = model.fit(x_train, y_train, epochs=30, batch_size=32, 
        validation_split=0.2, callbacks=early_stop)

    print("After data augmentation on the perturbed test set")
    model.evaluate(x_test_pert, y_test_pert, batch_size=32)

    
