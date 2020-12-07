import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import cv2

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class PongDQN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, hidden=256, batch_size=64):
        super(PongDQN, self).__init__()
        self.linear_input_dim = self.linear_input(state_space_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inputs = state_space_dim[0] * state_space_dim[1]
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.fc1 = torch.nn.Linear(self.linear_input_dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, action_space_dim)
        self.initialize()
        my_layers = [self.conv1, self.conv2, self.conv3, self.fc1, self.fc2]
        for l in my_layers:
            l.to(self.device)
        print("NN device", self.device)

    def linear_input(self, state_space_dim):
        a = self.conv2d_dims(50, 8, 4)
        a = self.conv2d_dims(a, 4, 2)
        a = self.conv2d_dims(a, 3, 1)

        b = self.conv2d_dims(50, 8, 4)
        b = self.conv2d_dims(b, 4, 2)
        b = self.conv2d_dims(b, 3, 1)
        return a * b * 32

    def conv2d_dims(self, input, kernel_size, stride):
        return (input - (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.linear_input_dim)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def initialize(self):
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)


class DQNAgent(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = "AVe"  # A-letta & Ve-ronika hehe
        self.n_actions = 3
        self.state_space = (50, 50)
        self.batch_size = 100
        self.hidden = 32
        self.policy_net = PongDQN(self.state_space, self.n_actions, self.hidden, self.batch_size)
        self.target_net = PongDQN(self.state_space, self.n_actions, self.hidden, self.batch_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = ReplayMemory(50000)
        self.gamma = 0.99
        self.prev_1 = np.zeros((50, 50))
        self.prev_2 = np.zeros((50, 50))
        print("Agent device", self.device)

    def update_network(self, updates=1):
        for _ in range(updates):
            self._do_network_update()

    def _do_network_update(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))
        non_final_mask = 1 - torch.tensor(batch.done, dtype=torch.uint8)
        non_final_next_states = [s for nonfinal, s in zip(non_final_mask,
                                                          batch.next_state) if nonfinal > 0]
        non_final_next_states = torch.stack(non_final_next_states).to(self.device)
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        state_action_values = self.policy_net(state_batch
                                              .reshape(-1, 3, 50, 50)).gather(1, action_batch).to(self.device)

        next_state_values = torch.zeros(self.batch_size).to(self.device)
        next_state_values[non_final_mask.bool()] = self.target_net(non_final_next_states
                                                                   .reshape(-1, 3, 50, 50)).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = reward_batch + self.gamma * next_state_values

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1e-1, 1e-1)
        self.optimizer.step()

    def get_action(self, state, epsilon=-1):
        state, _ = self.preprocess(observation=state)
        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():
                state = torch.reshape(state, (-1, 3, 50, 50))
                # state = torch.from_numpy(state.reshape(-1, 3, 50, 50)).float()
                q_values = self.policy_net(state)
                #print(q_values)
                action = torch.argmax(q_values).item()
                #print("Choosing", action)
                return action
        else:
            action = random.randrange(self.n_actions)
            return action


    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, state, action, next_state, reward, done):
        action = torch.Tensor([[action]]).long()
        reward = torch.tensor([reward], dtype=torch.float32)
        stacked_state, state = self.preprocess(state)
        stacked_next_state, next_state = self.preprocess(next_state, state, self.prev_1)

        self.prev_2 = self.prev_1
        self.prev_1 = state

        self.memory.push(stacked_state, action, stacked_next_state, reward/10, done)

    def preprocess(self, observation, prev_1=None, prev_2=None):

        #observation = observation[::4, ::4, 0]  # downsample by factor of 4
        #observation[observation == 33] = 0  # erase background
        #observation[(observation == 75) | (observation == 202)] = 1
        #observation[(observation == 255)] = 2

        observation = cv2.resize(observation, (int(50), int(50))).mean(axis=-1)
        observation[observation < 50] = 0  # erase background
        observation[(observation == 255)] = 2
        observation[(observation != 0) & (observation != 2)] = 1

        # convert the background in black and the players and the paddle in white
        #observation[observation == 33] = 0  # erase background
        #observation[(observation == 75) | (observation == 202)] = 1
        #observation[(observation == 255)] = 2

        #observation = np.array(observation)
        #observation = cv2.resize(observation, (int(50), int(50))).mean(axis=-1)

        if prev_1 is None:
            prev_1 = observation
        if prev_2 is None:
            prev_2 = observation

        observation = observation.reshape(50, 50, 1)
        prev_1 = prev_1.reshape(50, 50, 1)
        prev_2 = prev_2.reshape(50, 50, 1)

        stacked = np.concatenate((prev_1, prev_2, observation), axis=-1)
        stacked = torch.from_numpy(stacked).float().to(self.device)
        return stacked, observation

    def load_model(self):
        weights = torch.load("model.mdl", map_location=torch.device(self.device))
        self.policy_net.load_state_dict(weights, strict=False)
        self.target_net.load_state_dict(weights, strict=False)
        self.target_net.eval()

    def get_name(self):
        return self.name

    def reset(self):
        self.prev_1 = np.zeros((50, 50))
        self.prev_2 = np.zeros((50, 50))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)