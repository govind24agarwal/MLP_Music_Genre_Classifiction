import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras


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

        # 2nd hidden layer
        keras.layers.Dense(256, activation="relu"),

        # 3rd hidden layer
        keras.layers.Dense(64, activation="relu"),

        # output layer
        keras.layers.Dense(10, activation="softmax")
    ])

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
