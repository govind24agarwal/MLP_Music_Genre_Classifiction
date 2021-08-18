import json
import numpy as np

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
