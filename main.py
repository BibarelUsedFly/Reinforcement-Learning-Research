import gymnasium as gym
import numpy as np

def compute_cummulative_rewards(mat):
    row_sums = np.sum(mat, axis=1, keepdims=True)
    
    return row_sums - np.cumsum(mat, axis=1)

env = gym.make("LunarLander-v2")
observation, info = env.reset(seed=42)


NUM_ACTIONS  = 1
NUM_STATES = 8
NUM_EPISODES = 50
NUM_TIMESTEPS  = 50

rewards = np.zeros((NUM_EPISODES, NUM_TIMESTEPS))
states = np.zeros((NUM_EPISODES, NUM_TIMESTEPS, NUM_STATES))
actions = np.zeros((NUM_EPISODES, NUM_TIMESTEPS, NUM_ACTIONS))



for episode in range(NUM_EPISODES):    
    # Generate episode
    for timestamp in range(NUM_TIMESTEPS):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        rewards[episode, timestamp] = reward
        states[episode, timestamp] = observation

        if timestamp == NUM_TIMESTEPS:
            # What happens after terminated?
            observation, info = env.reset()
            
            timestamp += 1

env.close()

cummulative_rewards = compute_cummulative_rewards(rewards)

# reshape matrices
cummulative_rewards = np.reshape(cummulative_rewards, (NUM_EPISODES * NUM_TIMESTEPS, -1))
states = np.reshape(states, (NUM_EPISODES * NUM_TIMESTEPS, -1))
actions = np.reshape(actions, (NUM_EPISODES * NUM_TIMESTEPS, -1))

X = np.hstack((states, cummulative_rewards))
y = actions

print("Features")
print(X)

print("Classes")
print(y)
