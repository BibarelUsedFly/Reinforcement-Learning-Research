import numpy as np

def compute_cummulative_rewards(mat):
    row_sums = np.sum(mat, axis=1, keepdims=True)
    
    return row_sums - np.cumsum(mat, axis=1)


def generate_dataset(num_actions,
                     num_states,
                     num_episodes,
                     num_timesteps,
                     env
                     ):

    observation, _ = env.reset(seed=42)

    all_X = []
    all_y = []

    for _ in range(num_episodes):
        rewards = np.zeros((num_timesteps, 1))
        states = np.zeros((num_timesteps, num_states))
        actions = np.zeros((num_timesteps, num_actions))

        episode_length = 0  # Initialize episode length

        # Generate episode
        for timestamp in range(num_timesteps):
            action = env.action_space.sample()
            observation, reward, terminated, _, _ = env.step(action)

            rewards[timestamp] = reward
            states[timestamp] = observation

            episode_length = timestamp + 1

            if terminated:
                observation, _ = env.reset()
                break

    cummulative_rewards = compute_cummulative_rewards(rewards)

    # Trim matrices using the actual episode length
    rewards = rewards[:episode_length]
    states = states[:episode_length]
    actions = actions[:episode_length]

    # Reshape matrices
    cummulative_rewards = np.reshape(cummulative_rewards[:episode_length], (-1, 1))
    states = np.reshape(states, (-1, num_states))
    actions = np.reshape(actions, (-1, num_actions))

    curr_X = np.hstack((states, cummulative_rewards))
    curr_y = actions

    all_X.append(curr_X)
    all_y.append(curr_y)

    env.close()

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    return (X, y)