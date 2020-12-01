import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden) #50*50, 64
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space) # 64,3
        self.value_layer = torch.nn.Linear(self.hidden, 1)
        self.sigma = torch.nn.Parameter(torch.tensor([10.]))  # torch.tensor([5.])  # TODO: Implement accordingly (T1, T2) -- DONE T1
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def sigmoid(self,x):
        return 1.0 / (1.0 + np.exp(-x.detach().numpy()))  # sigmoid "squashing" function to interval [0,1]

    def forward(self, x, variance):
        x = self.fc1(x)
        x = F.relu(x)
        mu = self.fc2_mean(x)
        sigma = variance  # TODO: Is it a good idea to leave it like this? -- DONE
        # TODO: Instantiate and return a normal distribution -- DONE
        # with mean mu and std of sigma (T1)
        normal_dist = Normal(mu, torch.sqrt(sigma))

        # TODO: Add a layer for state value calculation (T3) -- DONE
        value = self.value_layer(x)
        out = self.sigmoid(value)
        return normal_dist, out



class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.baseline = 20
        self.variance = self.policy.sigma
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.values = []

    def episode_finished(self, episode_number):
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        values = torch.stack(self.values, dim=0).to(self.train_device).squeeze(-1)
        self.states, self.action_probs, self.rewards, self.values = [], [], [], []

        # TODO: Update policy variance (T2) -- DONE
        c = 5e-4
        # self.variance = self.policy.sigma * np.exp(-c * episode_number)  # Exponentially decaying variance

        # TODO: Compute discounted rewards (use the discount_rewards function) -- DONE
        rewards = discount_rewards(rewards, gamma=self.gamma)
        rewards = (rewards - torch.mean(rewards))/torch.std(rewards)  # REINFORCE with normalized rewards

        # TODO: Compute critic loss and advantages (T3) -- DONE
        advantages = rewards - values

        # TODO: Compute the optimization term (T1, T3) -- DONE
        loss = torch.sum(-action_probs * advantages.detach())  # Actor critic
        actor_loss = loss.mean()
        critic_loss = advantages.pow(2).mean()
        actor_critic_loss = actor_loss + critic_loss
        # loss = torch.sum(-rewards * action_probs)  # REINFORCE
        # loss = torch.sum(-(rewards - self.baseline) * action_probs)  # REINFORCE with baseline

        # TODO: Compute the gradients of loss w.r.t. network parameters (T1) -- DONE
        actor_critic_loss.backward()
        # loss.backward()

        # TODO: Update network parameters using self.optimizer and zero gradients (T1) -- DONE
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, evaluation=False):
        observation = preprocess(observation)
        x = torch.from_numpy(observation).float().to(self.train_device)

        # TODO: Pass state x through the policy network (T1) -- DONE
        _, aprob = self.policy.forward(x, self.variance)

        action = 1 if np.random.uniform() < aprob else 2  # roll the dice!
        return action
        #
        # # TODO: Return mean if evaluation, else sample from the distribution returned by the policy (T1) -- DONE
        # if evaluation:
        #     action = actions_distribution.mean
        # else:
        #     action = actions_distribution.sample((1,))[0]
        #
        # # TODO: Calculate the log probability of the action (T1) -- DONE
        # act_log_prob = actions_distribution.log_prob(action)
        #
        # # TODO: Return state value prediction, and/or save it somewhere (T3) -- DONE
        # self.values.append(value)
        #
        # return action, act_log_prob

    def store_outcome(self, observation, action_prob, action_taken, reward):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))

def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r
def preprocess(observation, previous_observation=None):
    observation = observation[::4, ::4, 0]  # downsample by factor of 4

    # keep the ball color at 255
    observation[observation == 33] = 0  # erase background (background type 2)
    observation[(observation == 75) | (observation == 202)] = 50  # join color for the players

    # show after color encoding
    # plt.imshow(observation)
    # plt.colorbar()
    # plt.title("Observation after downsampling and color encoding")
    # plt.show()

    # memorize the previous state
    subtract_next = observation

    previous_observation = np.asarray(previous_observation)
    if previous_observation.any():
        observation = observation - 0.5 * previous_observation

    # Show after substraction
    # plt.imshow(observation)
    # plt.colorbar()
    # plt.title("Observation after subtraction of previous image")
    # plt.show()
    observation = observation.astype(np.float).ravel()
    return observation#, subtract_next
