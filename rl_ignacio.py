from red_neuronal import NeuralNetwork 
import numpy as np

import gymnasium as gym
env = gym.make("Pendulum-v1")


k = 10
num_episodios = 100

total_reward = 0
terminated = False
truncated = False
max_steps = 100

training_set = []

net = NeuralNetwork(2, 10, 1)

state, info = env.reset(seed=42)

def calculate_total_reward(states, rewards, actions, context_lenght, gamma=0.1):
    state_to_use = states[-context_lenght]
    rewards_to_use = rewards[-context_lenght:]
    action_to_use = actions[-context_lenght]
    total_reward = 0
    for i in range(len(rewards_to_use)):
        total_reward += rewards_to_use[i] * gamma**(i)
    
    return state_to_use, total_reward, action_to_use


for _ in range(num_episodios):
    steps = 0
    episode_reward = 0
    episode_rewards = []
    episode_states = []
    episode_actions = []
    state, info = env.reset()
    while (steps < max_steps):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        
        episode_rewards.append(reward)
        episode_states.append(state)
        episode_actions.append(action)

        episode_reward += reward

        steps += 1
        if terminated:
            break
    
    training_set.append(calculate_total_reward(episode_states, episode_rewards, episode_actions, k))
    
    total_reward += episode_reward/steps

env.close()

X = []
Y = []

for episode in training_set:
    X.append(np.append(episode[0], episode[1]))
    Y.append(episode[2])


# train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)



print(X)
print(Y)
print(total_reward)