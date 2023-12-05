import json
import numpy as np
import torch
import torch.functional as F


def load_parameters(file_path):
    with open(file_path, "r") as file:
        parameters = json.load(file)
    return parameters


def save_dataset(X, y, file_path):
    np.savez(file_path, X=X, y=y)


def load_dataset(file_path):
    data = np.load(file_path)
    return data["X"], data["y"]


def get_nn_action(observation, model):

    X = np.hstack((observation, 1.5))
    X = torch.from_numpy(X).float()
    y = F.softmax(model(X), dim=0)
    y = torch.argmax(y)
    action = y.item()

    return action


# TODO


def get_agent_action(agent):
    pass
