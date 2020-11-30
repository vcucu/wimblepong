import torch
import torch.nn.functional as F
import wimblepong
import numpy as np
from torch.distributions import Normal


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.H = 50  # number of hidden layer neurons
        self.batch_size = 50  # every how many episodes to do a param update?
        self.learning_rate = 1e-4
        self.gamma = 0.99  # discount factor for reward
        self.decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
        self.resume = False  # resume from previous checkpoint?
        self.render = False

        self.D = 50 *50  # input dimensionality: 50x50 grid
        self.init_model_wights()

    def sigmoid(self,x):
        return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]

    def init_model_wights(self):
        # model initialization
        self.model = {}
        self.model['W1'] = np.random.randn(self.H, self.D) / np.sqrt(self.D)  # "Xavier" initialization
        self.model['W2'] = np.random.randn(self.H) / np.sqrt(self.H)

        self.grad_buffer = {k: np.zeros_like(v) for k, v in
                            self.model.items()}  # update buffers that add up gradients over a batch
        self.rmsprop_cache = {k: np.zeros_like(v) for k, v in
                              self.model.items()}  # rmsprop memory

    def policy_forward(self,x):
        h = np.dot(self.model['W1'], x)
        h[h < 0] = 0  # ReLU nonlinearity
        logp = np.dot(self.model['W2'], h)
        p = self.sigmoid(logp)
        return p, h  # return probability of taking action 2, and hidden state

    def policy_backward(self, eph, epdlogp,epx):
        """ backward pass. (eph is array of intermediate hidden states) """
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, self.model['W2'])
        dh[eph <= 0] = 0  # backpro prelu
        dW1 = np.dot(dh.T, epx)
        return {'W1': dW1, 'W2': dW2}