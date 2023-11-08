import gymnasium as gym
from neural_net import DiscreteNet

from dataset_generator import generate_dataset

env = gym.make("LunarLander-v2")
observation, info = env.reset(seed=42)

NUM_ACTIONS  = 1
NUM_STATES = 8
NUM_EPISODES = 50
NUM_TIMESTEPS  = 200

INPUT_DIM = NUM_ACTIONS + NUM_STATES
OUTPUT_DIM = 4

if __name__ == "__main__":
    X, y  = generate_dataset(NUM_ACTIONS, NUM_STATES, NUM_EPISODES, NUM_TIMESTEPS, env)
    net = DiscreteNet(INPUT_DIM, OUTPUT_DIM)