import argparse
import sys
import gym
import wimblepong
import pickle as pickle
import copy
import numpy as np
import PIL.Image as Image
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn as nn


def parse_args(args=sys.argv[1:]):
    # TODO [nice-to-have] lag continue training taking a file of weights already pretrained
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Directory to agent 1 to be tested.", default="agents/PG")
    parser.add_argument("--env", type=str, default="WimblepongVisualSimpleAI-v0",
                        help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=50000,
                        help="Number of episodes to train for")
    parser.add_argument("--print_stats", type=bool, default=True)
    parser.add_argument("--run_id", type=int, default=0)

    return parser.parse_args(args)


class PolicyNetwork(nn.Module):
    """ Simple two-layer MLP for policy network. """

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)

        x = self.fc2(x)
        prob_up = torch.sigmoid(x)

        return prob_up


# Entry point of the script
def main(args):
    # environment
    # env = gym.make("Pong-v0")
    env = gym.make("WimblepongVisualSimpleAI-v0")
    num_episodes = args.train_episodes
    # hyperparameters
    H = 200  # number of hidden layer neurons
    batch_size = 30  # every how many episodes to do a param update?
    learning_rate = 1e-4  # for convergence (too low- slow to converge, too high,never converge)
    gamma = 0.9  # discount factor for reward (i.e later rewards are exponentially less important)
    decay_rate = 0.89  # decay factor for RMSProp leaky sum of grad^2
    resume = False  # resume from previous checkpoint?

    def prepro(I):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = I[::4, ::4, 0]  # downsample by factor of 2
        I[I == 33] = 0  # erase background
        I[(I == 75) | (I == 202) | (I == 255)] = 1

        return I.astype(np.float).ravel()  # flattens

    # model initialization
    D = 50 * 50  # input dimensionality: 80x80 grid (the pong world)
    if resume:
        model = pickle.load(open('save.p', 'rb'))  # load from pickled checkpoint
    else:
        model = PolicyNetwork(D, H)
        model = {}  # initialize model
        # rand returns a sample (or samples) from the standard normal distribution
        # xavier algo determines the scale of initialization based on the number of input and output neurons.
        # Imagine that your weights are initially very close to 0. What happens is that the signals shrink as it goes through each
        # layer until it becomes too tiny to be useful. Now if your weights are too big, the signals grow at each layer
        # it passes through until it is too massive to be useful.
        # By using Xavier initialization, we make sure that the weights are not too small but not too big to propagate accurately the signals.
    # model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
    # model['W2'] = np.random.randn(H) / np.sqrt(H)
    # zeros like returns an array of zeros with the same shape and type as a given array.
    # we will update buffers that add up gradients over a batch
    # where the model contains kv pairs, weights layers etc
    grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
    ## rmsprop (gradient descent) memory used to update model
    rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}

    # activation function
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]

    def discount_rewards(r):
        """ take 1D float array of rewards and compute discounted reward """
        # initilize discount reward matrix as empty
        discounted_r = np.zeros_like(r)
        # to store reward sums
        running_add = 0
        # for each reward
        for t in reversed(range(0, r.size)):
            # if reward at index t is nonzero, reset the sum, since this was a game boundary (pong specific!)
            if r[t] != 0: running_add = 0
            # increment the sum
            # https://github.com/hunkim/ReinforcementZeroToAll/issues/1
            running_add = running_add * gamma + r[t]
            # earlier rewards given more value over time
            # assign the calculated sum to our discounted reward matrix
            discounted_r[t] = running_add
        return discounted_r

    # forward propagation via numpy woot!
    def policy_forward(x):
        # matrix multiply input by the first set of weights to get hidden state
        # will be able to detect various game scenarios (e.g. the ball is in the top, and our paddle is in the middle)
        h = np.dot(model['W1'], x)
        # apply an activation function to it
        # f(x)=max(0,x) take max value, if less than 0, use 0
        h[h < 0] = 0  # ReLU nonlinearity
        # repeat process once more
        # will decide if in each case we should be going UP or DOWN.
        logp = np.dot(model['W2'], h)
        # squash it with an activation (this time sigmoid to output probabilities)
        p = sigmoid(logp)
        return p, h  # return probability of taking action 2, and hidden state

    def policy_backward(eph, epdlogp):
        """ backward pass. (eph is array of intermediate hidden states) """
        # recursively compute error derivatives for both layers, this is the chain rule
        # epdlopgp modulates the gradient with advantage
        # compute updated derivative with respect to weight 2. It's the parameter hidden states transpose * gradient w/ advantage (then flatten with ravel())
        dW2 = np.dot(eph.T, epdlogp).ravel()
        # Compute derivative hidden. It's the outer product of gradient w/ advatange and weight matrix 2 of 2
        dh = np.outer(epdlogp, model['W2'])
        # apply activation
        dh[eph <= 0] = 0  # backpro prelu
        # compute derivative with respect to weight 1 using hidden states transpose and input observation
        dW1 = np.dot(dh.T, epx)
        # return both derivatives to update weights
        return {'W1': dW1, 'W2': dW2}

    # Each timestep, the agent chooses an action, and the environment returns an observation and a reward.
    # The process gets started by calling reset, which returns an initial observation
    observation = env.reset()
    prev_x = None  # used in computing the difference frame
    # observation, hidden state, gradient, reward
    xs, hs, dlogps, drs = [], [], [], []
    # current reward
    running_reward = None
    # sum rewards
    reward_sum = 0
    # where are we?
    episode_number = 0
    cumulative_rewards = []

    # begin training!
    while episode_number < num_episodes:

        # preprocess the observation, set input to network to be difference image
        # Since we want our policy network to detect motion
        # difference image = subtraction of current and last frame
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x
        # so x is our image difference, feed it in!

        # forward the policy network and sample an action from the returned probability
        aprob, h = policy_forward(x)
        # this is the stochastic part
        # since not apart of the model, model is easily differentiable
        # if it was apart of the model, we'd have to use a reparametrization trick (a la variational autoencoders. so badass)
        action = 2 if np.random.uniform() < aprob else 3  # roll the dice!

        # record various intermediates (needed later for backprop)
        xs.append(x)  # observation
        hs.append(h)  # hidden state
        y = 1 if action == 2 else 0  # a "fake label"
        dlogps.append(
            y - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

        # step the environment and get new measurements
        # env.render()
        observation, reward, done, info = env.step(action)
        reward_sum += reward

        drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

        if done:  # an episode finished
            episode_number += 1

            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            # each episode is a few dozen games
            epx = np.vstack(xs)  # obsveration
            eph = np.vstack(hs)  # hidden
            epdlogp = np.vstack(dlogps)  # gradient
            epr = np.vstack(drs)  # reward
            xs, hs, dlogps, drs = [], [], [], []  # reset array memory

            # the strength with which we encourage a sampled action is the weighted sum of all rewards afterwards, but later rewards are exponentially less important
            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr = discounted_epr - np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            # advatnage - quantity which describes how good the action is compared to the average of all the action.
            epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
            grad = policy_backward(eph, epdlogp)
            for k in model: grad_buffer[k] += grad[k]  # accumulate grad over batch

            # perform rmsprop parameter update every batch_size episodes
            if episode_number % batch_size == 0:
                for k, v in model.items():
                    g = grad_buffer[k]  # gradient
                    rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                    model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                    grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

            # boring book-keeping
            cumulative_rewards.append(reward_sum)

            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('Episode %d reward total was %f. running mean: %f' % (episode_number, reward_sum, running_reward))
            if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
            reward_sum = 0
            observation = env.reset()  # reset env
            prev_x = None
            if episode_number == 10000 or episode_number == 20000 or episode_number == 35000 or episode_number == 55000 or episode_number == 75000 or episode_number == 90000:
                plot_rewards(cumulative_rewards, episode_number)

    plot_rewards(cumulative_rewards, 0)

    # if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
    #     print('ep %d: game finished, reward: %f' % (episode_number, reward)) + (' ' if reward == -1 else ' !!!!!!!!')


def plot_rewards(rewards, index):
    plt.figure(2)
    plt.clf()
    rewards_t = torch.tensor(rewards, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative reward')
    plt.grid(True)
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    # plt.savefig('train_plot_%.png')
    plt.savefig('train_plot_train_pg_{}.png'.format(index), format='png')

    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)
