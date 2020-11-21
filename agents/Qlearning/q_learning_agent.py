import torch
import torch.nn.functional as F
import wimblepong
import numpy as np
import pandas as pd
import gym
import random
import sys
import matplotlib.pyplot as plt

import wimblepong


class Agent():
    def __init__(self, env, player_id=1):
        #Params
        gamma = 0.98
        alpha = 0.1
        target_eps = 0.1
        initial_q = 0
        actions_num = 3

        self.env = env
        self.player_id = player_id # player id that determines on which side do we play on
        self.name = "Qlearning Agent"
        self.q_grid = np.zeros((200, 200, 3, actions_num)) + initial_q

    def get_name(self):
        return self.name

    def get_action(self, ob = None):
        #TODO load q_grid from memory if not loaded atm
        max_q = np.argmax(self.q_values[ob])
        return max_q

    def get_train_action(self, state, epsilon, actions_num):
        if np.random.random() <= epsilon:
            random_action = np.random.randint(0, actions_num)
            return random_action
        else:
            index = state
            max_q = np.argmax(self.q_grid[index])
            return max_q

    def update_q_value(self, old_state, action, new_state, reward, done):
        print(old_state) #This will need to be converted to the index
        old_cell_index = old_state
        new_cell_index = new_state
        max_next_reward = np.max(self.q_grid[new_cell_index])
        if done:
            new_value = self.q_grid[old_cell_index][action] + self.alpha * (reward - self.q_grid[old_cell_index][action])
        else:
            new_value = self.q_grid[old_cell_index][action] + self.alpha * (
                    reward + (self.gamma * max_next_reward) - self.q_grid[old_cell_index][action])
        self.q_grid[old_cell_index][action] = new_value

    def train(self, episodes, env, actions_num):
        #TODO somehow take into account which player are we

        # Training loop
        ep_lengths, epl_avg = [], []
        rewards_list, rewards_avg = [], []
        for ep in range(episodes): #+ test_episodes
            #test = ep > episodes
            state, done, steps = env.reset(), False, 0
            b = 11111  # 2222
            epsilon = b / (b + ep)  # 0.2
            cumulative_reward = 0
            while not done:
                action = self.get_train_action(state, epsilon, actions_num)
                # print("action",action)
                new_state, reward, done, _ = env.step(action)
                cumulative_reward += reward
                #if not test:
                self.update_q_value(state, action, new_state, reward, done)
                #else:
                #    env.render()
                state = new_state
                steps += 1
            ep_lengths.append(steps)
            epl_avg.append(np.mean(ep_lengths[max(0, ep - 500):]))
            rewards_list.append(cumulative_reward)
            rewards_avg.append(np.mean(rewards_list[max(0, ep - 500):]))
            if ep % 200 == 0:
                print("Episode {}, average timesteps: {:.2f}".format(ep, np.mean(ep_lengths[max(0, ep - 200):])))
        np.save("q_values.npy", self.q_grid)
        # Calculate the value function
        values = self.q_grid
        """values = values.max(axis = 4)"""
        values = values.max(axis=8)
        # values = np.zeros(q_grid.shape[:-1])
        np.save("value_func.npy", values)

        self.show_results()

    def show_results(self, rewards_list, rewards_avg, ep_lengths, epl_avg):

        # Plot the heatmap
        """values = values.mean(axis = 3)
        values = values.mean(axis = 1)
        seaborn.heatmap(values)
        plt.title("Location versus angle values heatmap")
        plt.show()"""

        plt.plot(rewards_list)
        plt.plot(rewards_avg)
        plt.title("Rewards in training")
        plt.show()

        # Draw plots
        plt.plot(ep_lengths)
        plt.plot(epl_avg)
        plt.legend(["Episode length", "500 episode average"])
        plt.title("Episode lengths")
        plt.show()

    def reset(self):
        # Nothing to done for now...
        return


