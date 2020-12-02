import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class PongDQN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, hidden=256, batch_size=64):
        super(PongDQN, self).__init__()
        self.linear_input_dim = self.linear_input(state_space_dim)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.inputs = state_space_dim[0] * state_space_dim[1]

        # state_space_dim[0] (3)
        self.conv1 = nn.Conv2d(1, hidden, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(hidden, 64, kernel_size=5, stride=1)
        self.fc1 = torch.nn.Linear(self.linear_input_dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, action_space_dim)
        self.initialize()

    def initialize(self):
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def linear_input(self, state_space_dim):
        a = self.conv2d_dims(50, 5, 1)
        a = self.conv2d_dims(a, 5, 1)
        b = self.conv2d_dims(50, 5, 1)
        b = self.conv2d_dims(b, 5, 1)
        return a * b * 64

    def conv2d_dims(self, input, kernel_size, stride):
        return (input - (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.linear_input_dim)
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        return x


class DQNAgent(object):
    def __init__(self, env_name, state_space, n_actions, replay_buffer_size,
                 batch_size, hidden_size, gamma):
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

        self.env_name = env_name
        self.n_actions = n_actions
        self.state_space_dim = state_space
        self.policy_net = PongDQN(state_space, n_actions, hidden_size, batch_size)
        self.target_net = PongDQN(state_space, n_actions, hidden_size, batch_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=5e-4)
        self.memory = ReplayMemory(replay_buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma

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
        non_final_next_states = torch.stack(non_final_next_states)
        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch
                                              .reshape(-1, 1, 50, 50)).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size).to(self.device)
        next_state_values[non_final_mask.bool()] = self.target_net(non_final_next_states
                                                                   .reshape(-1, 1, 50, 50)).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = reward_batch + self.gamma * next_state_values

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values.squeeze(),
                                expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1e-1, 1e-1)
        self.optimizer.step()

    def get_action(self, state, epsilon=0.05):
        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():
                state = torch.from_numpy(state).float()
                q_values = self.policy_net(state)
                return torch.argmax(q_values).item()
        else:
            return random.randrange(self.n_actions)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, state, action, next_state, reward, done):
        action = torch.Tensor([[action]]).long()
        reward = torch.tensor([reward], dtype=torch.float32)
        next_state = torch.from_numpy(next_state).float()
        state = torch.from_numpy(state).float()
        self.memory.push(state, action, next_state, reward, done)


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
