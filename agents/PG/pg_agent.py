import torch
import torch.nn.functional as F
import wimblepong
import numpy as np
from torch.distributions import Normal


class Agent(object):
    def __init__(self, env, policy,state_space, player_id=1):

        #Params
        self.reinforce_version = "1c" #options: 1a, 1b, 1c corresponding to the version of REINFORCE algorithm
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.prev_x = None  # used in computing the difference frame
        self.xs, self.hs, self.dlogps, self.drs = [], [], [], []
        self.state_space_dim = state_space

        self.env = env
        self.player_id = player_id
        self.name = "PG Agent"

    def get_name(self):
        return self.name

    def episode_finished(self, episode_number):
        # stack together all inputs, hidden states, action gradients, and rewards for this episode

        epx = np.vstack(self.xs)
        eph = np.vstack(self.hs)
        epdlogp = np.vstack(self.dlogps)
        epr = np.vstack(self.drs)
        self.xs, self.hs, self.dlogps, self.drs = [], [], [], []  # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = self.discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr = discounted_epr -  np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
        grad = self.policy.policy_backward(eph, epdlogp, epx)
        for k in self.policy.model: self.policy.grad_buffer[k] += grad[k]  # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % self.policy.batch_size == 0:
            for k, v in self.policy.model.items():
                g = self.policy.grad_buffer[k]  # gradient
                self.policy.rmsprop_cache[k] = self.policy.decay_rate * self.policy.rmsprop_cache[k] + (1 - self.policy.decay_rate) * g ** 2
                self.policy.model[k] += self.policy.learning_rate * g / (np.sqrt(self.policy.rmsprop_cache[k]) + 1e-5)
                self.policy.grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

        # boring book-keeping

        self.prev_x = np.zeros(50*50)

    def get_action(self, observation):

        # forward the policy network and sample an action from the returned probability
        aprob, h = self.policy.policy_forward(observation)
        action = 2 if np.random.uniform() < aprob.any() else 3  # roll the dice!

        # record various intermediates (needed for backprop)
        self.xs.append(observation)  # observation
        self.hs.append(h)  # hidden state
        y = 1 if action == 2 else 0  # a "fake label"
        self.dlogps.append(y - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

        return action, aprob

    def preprocess(self,observation, previous_observation=None):
        # TODO do stuffs (or alternatively we can implement it outside of this module)

        observation = observation[::4, ::4, 0]  # downsample by factor of 4

        # keep the ball color at 255
        observation[observation == 33] = 0  # erase background (background type 2)
        observation[(observation == 75) | (observation == 202)] = 50  # join color for the players
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
        return observation, subtract_next



      #  return observation.astype(np.float).ravel()

    def discount_rewards(self,r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * self.policy.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def store_outcome(self, observation, action_prob, action_taken, reward):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))

    def reset(self):
        return


