import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers.core import Dropout

# path to json file that stores MFCCs and genre labels for each processed segment
DATA_PATH = "/home/govind/Documents/ML/Velario Youtube/extracting_mfccs_music_genre/data.json"


def load_data(data_path):
    """load training datasetfrom json file

    Args:
        data_path (String)): path to json file containing data

    Returns:
        x (ndarray): mfcc data of music dataset segments, Training inputs
        y (ndarray): label data of mfccs, Training targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    x = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return x, y


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs

    Args:
        history (object): Training history of the model
    """

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error subplot
    axs[1].plot(history.history["loss"], label='train error')
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Eroor")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


if __name__ == "__main__":

    # load data
    x, y = load_data(DATA_PATH)

    # create train/test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # build network  topology
    model = keras.Sequential([

        # input layer
        keras.layers.Flatten(input_shape=(x.shape[1], x.shape[2])),

        # 1st hidden layer
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dropout(0.05),

        # 2nd hidden layer
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.05),

        # 3rd hidden layer
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.05),

        # output layer
        keras.layers.Dense(10, activation="softmax")
    ])

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

# train model
history = model.fit(x_train,
                    y_train,
                    validation_data=(x_test, y_test),
                    batch_size=32,
                    epochs=50)

# plot accuracy and error as a function of the epochs
plot_history(history)
