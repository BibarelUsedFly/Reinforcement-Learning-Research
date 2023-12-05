import json
import numpy as np


def load_parameters(file_path):
    with open(file_path, "r") as file:
        parameters = json.load(file)
    return parameters


def save_dataset(X, y, file_path):
    np.savez(file_path, X=X, y=y)


def load_dataset(file_path):
    data = np.load(file_path)
    return data["X"], data["y"]
