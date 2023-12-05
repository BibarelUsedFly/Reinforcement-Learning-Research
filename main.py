from utils import load_parameters
import gymnasium as gym
from neural_net import DiscreteNet
import torch
import torch.nn.functional as F
import numpy as np

config_file_path = "config.json"
parameters = load_parameters(config_file_path)

NUM_EPISODES = parameters["NUM_EPISODES"]
NUM_TIMESTEPS = parameters["NUM_TIMESTEPS"]
INPUT_DIM = parameters["INPUT_DIM"]
OUTPUT_DIM = parameters["OUTPUT_DIM"]
NUM_EPOCHS = parameters["NUM_EPOCHS"]
WEIGHTS_PATH = parameters["WEIGHTS_PATH"]
DROPOUT_RATE = parameters["DROPOUT_RATE"]


env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)

model = DiscreteNet(INPUT_DIM, OUTPUT_DIM, DROPOUT_RATE)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device("cpu")))
model.eval()

try:
    for _ in range(NUM_EPISODES):

        episode_length = 0

        action = env.action_space.sample()

        for timestamp in range(NUM_TIMESTEPS):

            observation, reward, terminated, _, _ = env.step(action)
            X = np.hstack((observation, 10))
            X = torch.from_numpy(X).float()

            y = F.softmax(model(X), dim=0)
            y = torch.argmax(y)
            action = y.item()

            episode_length = timestamp + 1

            if terminated:
                observation, _ = env.reset()
                break
except KeyboardInterrupt:
    print("Exiting...")
