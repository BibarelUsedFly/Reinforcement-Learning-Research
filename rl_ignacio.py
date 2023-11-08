from red_neuronal import Net, train_net
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch


import gymnasium as gym
env = gym.make("CartPole-v1")


k = 10
num_episodios = 100

total_reward = 0
terminated = False
truncated = False
max_steps = 100

training_set = []

state, info = env.reset(seed=42)

def calculate_total_reward(states, rewards, actions, context_lenght, gamma=0.1):
    if len(rewards) < context_lenght:
        context_lenght = len(rewards)
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

print('Finished Dataset build')


tensor_x = torch.Tensor(np.array(X)) # transform to torch tensor
tensor_y = torch.Tensor(np.array(Y))

my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset


input_dim = len(state) + 1
output_dim = 2

net = Net(input_dim, 2)

net = train_net(net, my_dataset)

print('Finished Training')


state, info = env.reset()
test_vector = torch.Tensor(np.append(state, 10))
# print(np.array(state, 100))
predict = net(test_vector)
print(predict)


# print(X)
# print(Y)
# print(total_reward)