import torch
from torch import nn
from collections import namedtuple
import random
import numpy as np
import cv2
from torch.autograd.grad_mode import F
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from collections import deque
import PIL.Image as Image


class Network(nn.Module):
    def __init__(self, in_channels=4):
        super(Network, self).__init__()
        self.feature_dim = 512

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        nn.init.kaiming_normal_(self.conv1.weight)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        nn.init.kaiming_normal_(self.conv2.weight)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        nn.init.kaiming_normal_(self.conv3.weight)

        self.fc4 = nn.Linear(7 * 7 * 64, self.feature_dim)
        nn.init.kaiming_normal_(self.fc4.weight)

        self.out = nn.Linear(self.feature_dim, 3)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))

        x = self.relu(self.conv2(x))

        x = self.relu(self.conv3(x))

        x = x.view(x.shape[0], -1)

        x = self.relu(self.fc4(x))

        out = self.out(x)
        return out


device = "cpu"


class PongDQN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, hidden=100, batch_size=64):
        super(PongDQN, self).__init__()
        self.inputs = state_space_dim
        state_space_dim = self.inputs  # batch_size * inputs *hidden
        self.fc1 = nn.Linear(state_space_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_space_dim)

    def forward(self, x):
        x = x.view(-1, self.inputs)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent(object):
    def __init__(self, player_id=1):
        self.network = Network().to(device)
        self.target_network = Network().to(device)
        self.player_id = player_id
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr=0.0001)

        self.memory = ReplayMemory(40000)
        self.batch_size = 32
        self.frame_number = 4
        self.frames = np.zeros((50, 50, self.frame_number))
        self.start = True
        self.policy_net = PongDQN(50 * 50, 3, 32, 32)
        self.epsilon = 0.0005
        self.discount = .99
        self.random_start_iter = 0
        self.params = self.network.parameters()

    def update_network(self):
        """
        Optimize the network accordingly with the experience
        """
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = ~torch.tensor(batch.done, dtype=torch.bool)
        non_final_next_states = [s for nonfinal, s in zip(non_final_mask,
                                                          batch.next_frame) if nonfinal > 0]
        non_final_next_states = torch.stack(non_final_next_states).float().to(device)
        state_batch = torch.stack(batch.frame).float().to(device)
        action_batch = torch.cat(batch.action).long().to(device)
        reward_batch = torch.cat(batch.reward).float().to(device)

        actions = torch.nn.functional.one_hot(action_batch.view(-1), 3)

        state_action_values = (self.network(state_batch / 255.0) * actions).sum(1)  # .gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.target_network(non_final_next_states / 255.0).max(1)[0].detach()

        expected_state_action_values = reward_batch + self.discount * next_state_values

        self.optimizer.zero_grad()

        loss = nn.functional.smooth_l1_loss(state_action_values.squeeze(),
                                            expected_state_action_values)

        loss.backward()

        self.optimizer.step()

    def store_memory(self, state, action, next_state, reward, done):
        """
        Method to store in memory an specific observation, action, reward to be used later on
        """

        img = Image.fromarray(next_state)
        img = img.convert("L")
        next_state = img.resize((84, 84), Image.NEAREST)

        next_state = np.reshape(next_state, (1, 84, 84))

        action = torch.tensor([[action]], dtype=torch.int8)
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1
        reward = torch.tensor([reward], dtype=torch.int8)

        next_state = torch.from_numpy(
            np.append(next_state, self.frames[:(self.frame_number - 1), :, :], axis=0)).type(torch.uint8)
        frames = torch.from_numpy(self.frames).short().type(torch.uint8)

        self.memory.push(frames, action, next_state, reward, done)

    def load_model(self):
        """
        Loads a trained model from a file
        """
        weights = torch.load("model.mdl")
        self.network.load_state_dict(weights, strict=False)

    def get_action(self, observation) -> int:
        """
        Runs the network and selects an action to perform
        :return: the action to be performed
        """

        img = Image.fromarray(observation)
        img = img.convert("L")
        observation = img.resize((50, 50), Image.NEAREST)
        if self.start:
            observation = np.reshape(observation, (50, 50))
            self.start = False
            self.frames = np.stack((observation, observation, observation, observation), axis=0)
        else:
            observation = np.reshape(observation, (1, 50, 50))
            self.frames = np.append(observation, self.frames[:(self.frame_number - 1), :, :], axis=0)

        if self.random_start_iter > 0:
            self.random_start_iter -= 1
            action = np.random.randint(3)
        else:
            if np.random.random() < self.epsilon:
                action = random.randrange(3)
            else:
                # q_values = self.policy_net(observation)

                action = self.network(
                    torch.tensor(self.frames.reshape((1, self.frame_number, 50, 50)), dtype=torch.float32,
                                 device=device) / 255.0)
                _, action = torch.max(action, dim=1)
                action = int(action.item())
        return action

    def get_name(self) -> str:
        """
        Function to get the group name
        :return: group name
        """
        return "NETO_21"

    def reset(self):
        """
        Function to reset the agent state after each episode
        """
        self.frames = np.zeros((84, 84, self.frame_number))
        self.start = True
        self.appended_frame = 0

    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())


Transition = namedtuple('Transition',
                        ('frame', 'action', 'next_frame', 'reward', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.full_warning = True

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        else:
            if self.full_warning:
                print("Memory Full")
                self.full_warning = False
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
