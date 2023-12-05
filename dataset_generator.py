import numpy as np
from utils import save_dataset, load_parameters
import gymnasium as gym


config_file_path = "config.json"
parameters = load_parameters(config_file_path)

NUM_ACTIONS = parameters["NUM_ACTIONS"]
NUM_STATES = parameters["NUM_STATES"]
NUM_EPISODES = parameters["NUM_EPISODES"]
NUM_TIMESTEPS = parameters["NUM_TIMESTEPS"]
INPUT_DIM = parameters["INPUT_DIM"]
OUTPUT_DIM = parameters["OUTPUT_DIM"]
WEIGHTS_PATH = parameters["WEIGHTS_PATH"]
DATASET_PATH = parameters["DATASET_PATH"]


def compute_cumulative_rewards(arr):
    cum_sum = np.cumsum(arr[::-1])[::-1]

    return cum_sum


def generate_dataset(
    num_actions,
    num_states,
    num_episodes,
    num_timesteps,
    env,
    seed=42,
):

    observation, _ = env.reset(seed=seed)

    all_X = []
    all_y = []

    for _ in range(num_episodes):
        rewards = np.zeros((num_timesteps, 1))
        states = np.zeros((num_timesteps, num_states))
        actions = np.zeros((num_timesteps, num_actions))

        episode_length = 0

        for t in range(num_timesteps):
            action = env.action_space.sample()
            observation, reward, terminated, _, _ = env.step(action)

            rewards[t] = reward
            states[t] = observation
            actions[t] = action

            episode_length = t + 1

            if terminated:
                observation, _ = env.reset()
                break

        states = states[:episode_length]
        actions = actions[:episode_length]
        rewards = rewards[:episode_length]
        cummulative_rewards = compute_cumulative_rewards(rewards)

        states = np.reshape(states, (-1, num_states))
        actions = np.reshape(actions, (-1, num_actions))
        cummulative_rewards = np.reshape(cummulative_rewards, (-1, 1))

        curr_X = np.hstack((states, cummulative_rewards))
        curr_y = actions

        all_X.append(curr_X)
        all_y.append(curr_y)

    env.close()

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    return (X, y)


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")

    X, y = generate_dataset(
        NUM_ACTIONS,
        NUM_STATES,
        NUM_EPISODES,
        NUM_TIMESTEPS,
        env,
    )
    save_dataset(X, y, DATASET_PATH)
