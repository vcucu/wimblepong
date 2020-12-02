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

        self.conv1 = nn.Conv2d(3, hidden, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(hidden, 32, kernel_size=5, stride=1)
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
        return a * b * 32

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
    def __init__(self):
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        self.name = "AVe"  # A-letta & Ve-ronika hehe
        self.n_actions = 3
        self.state_space = (50, 50)
        self.batch_size = 256
        self.hidden = 64
        self.policy_net = PongDQN(self.state_space, self.n_actions, self.hidden, self.batch_size)
        self.target_net = PongDQN(self.state_space, self.n_actions, self.hidden, self.batch_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=5e-4)
        self.memory = ReplayMemory(50000)
        #self.glie_a = 5555
        self.gamma = 0.99
        self.prev_1 = np.zeros((50, 50))
        self.prev_2 = np.zeros((50, 50))

    def get_name(self):
        return self.name

    def reset(self):
        self.prev_1 = np.zeros((50, 50))
        self.prev_2 = np.zeros((50, 50))

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
                                              .reshape(-1, 3, 50, 50)).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size).to(self.device)
        next_state_values[non_final_mask.bool()] = self.target_net(non_final_next_states
                                                                   .reshape(-1, 3, 50, 50)).max(1)[0].detach()

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

    def get_action(self, state, epsilon=-1):
        state, _ = self.preprocess(state)
        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():
                state = torch.from_numpy(state.reshape(-1, 3, 50, 50)).float()
                q_values = self.policy_net(state)
                return torch.argmax(q_values).item()
        else:
            return random.randrange(self.n_actions)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, state, action, next_state, reward, done):
        action = torch.Tensor([[action]]).long()
        reward = torch.tensor([reward], dtype=torch.float32)
        stacked_state, state = self.preprocess(state)
        stacked_next_state, next_state = self.preprocess(next_state, state, self.prev_1)

        self.prev_2 = self.prev_1
        self.prev_1 = state

        #next_state = torch.from_numpy(next_state).float()  # are the two lines needed?
        #state = torch.from_numpy(state).float()  # are these two lines needed?
        self.memory.push(stacked_state, action, stacked_next_state, reward, done)

    def preprocess(self, observation, prev_1=None, prev_2=None):
        observation = observation[::4, ::4, 0]  # downsample by factor of 4
        observation[observation == 33] = 0  # erase background
        observation[(observation == 75) | (observation == 202) | (observation == 255)] = 1

        # show after color encoding
        # plt.imshow(observation)
        # plt.colorbar()
        # plt.title("Observation after downsampling and color encoding")
        # plt.show()
        # memorize the previous state
        # subtract_next = observation
        # previous_observation = np.asarray(previous_observation)
        # if previous_observation.any():
        #    observation = observation - 0.5 * previous_observation
        # observation = observation.astype(np.float)
        # Show after substraction
        # plt.imshow(observation)
        # plt.colorbar()
        # plt.title("Observation after subtraction of previous image")
        # plt.show()

        if prev_1 is None:
            prev_1 = self.prev_1
        if prev_2 is None:
            prev_2 = self.prev_2

        observation = torch.Tensor(observation)
        stacked = np.concatenate((self.prev_1, self.prev_2, observation), axis=-1)
        stacked = torch.from_numpy(stacked).float().unsqueeze(0)
        #stacked = stacked.transpose(1, 3)
        #print("4", stacked.shape)
        return stacked, observation


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
