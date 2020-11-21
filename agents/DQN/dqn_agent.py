import torch
import torch.nn.functional as F
import wimblepong
import numpy as np
import pandas as pd
import gym
import random
import sys
import matplotlib.pyplot as plt
from torch.distributions import Normal


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.sigma_version = '2b' # options '1', '2a', '2b'
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        self.sigma = 0
        self.init_sigma()
        self.init_weights()

    def init_sigma(self):
        if self.sigma_version == '1':
            self.sigma = torch.tensor([np.sqrt(5.)], dtype=torch.float32)
        elif self.sigma_version == '2a':
            self.sigma = torch.tensor([np.sqrt(10.)], dtype=torch.float32)
        elif self.sigma_version == '2b':
            self.sigma = torch.nn.Parameter(torch.tensor([np.sqrt(10.)], dtype=torch.float32))

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_mean = self.fc2_mean(x)
        sigma = self.sigma
        return Normal(action_mean, sigma)


class Agent(object):
    def __init__(self, env, policy, player_id=1):

        #Params
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.sigma = self.policy.sigma
        self.baseline = 20

        self.env = env
        self.player_id = player_id # player id that determines on which side do we play on
        self.name = "DQN Agent"

    def get_name(self):
        return self.name

    def get_action(self, ob = None):
        pass

    def get_train_action(self, state, epsilon, actions_num):
        pass

    def episode_finished(self, episode_number):
        action_probs = torch.stack(self.action_probs, dim=0) \
            .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        self.states, self.action_probs, self.rewards = [], [], []

        # (DONE) TODO: Compute discounted rewards (use the discount_rewards function)
        G = self.discount_rewards(rewards, self.gamma)

        # (DONE) TODO: Compute the optimization term (T1)
        T = len(rewards)
        gammas = torch.tensor([self.gamma ** t for t in range(T)]).to(self.train_device)
        if self.reinforce_version == '1a':  # REINFORCE
            loss = torch.sum(-gammas * G * action_probs)
        elif self.reinforce_version == '1b':  # REINFORCE with baseline
            loss = torch.sum(-gammas * (G - self.baseline) * action_probs)
        elif self.reinforce_version == '1c':  # REINFORCE with normalized discounted rewards
            G = (G - torch.mean(G))/torch.std(G)
            loss = torch.sum(-gammas * (G) * action_probs)

        # (DONE) TODO: Compute the gradients of loss w.r.t. network parameters (T1)
        loss.backward()

        # (DONE) TODO: Update network parameters using self.optimizer and zero gradients (T1)
        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.policy.sigma_version == '2a':
            self.policy.sigma = self.sigma * (np.e ** ((-5 * 10 ** (-4)) * episode_number))

    def get_action(self, observation, evaluation=False ):
        x = torch.from_numpy(observation).float().to(self.train_device)

        # (DONE) TODO: Pass state x through the policy network (T1)
        actions_distribution = self.policy.forward(x)

        # (DONE) TODO: Return mean if evaluation, else sample from the distribution
        if evaluation:
            action = actions_distribution.mean
        else:
            action = actions_distribution.sample()

        # (DONE) TODO: Calculate the log probability of the action (T1)

        #print(action[0], actions_distribution)
        act_log_prob = actions_distribution.log_prob(action[0])

        return action, act_log_prob

    def discount_rewards(r, gamma):
        discounted_r = torch.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size(-1))):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def store_outcome(self, observation, action_prob, action_taken, reward):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))

    def reset(self):
        # Nothing to done for now...
        return


