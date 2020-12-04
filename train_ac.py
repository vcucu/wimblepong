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


def parse_args(args=sys.argv[1:]):
    # TODO [nice-to-have] lag continue training taking a file of weights already pretrained
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Directory to agent 1 to be tested.", default="agents/PG")
    parser.add_argument("--env", type=str, default="WimblepongVisualSimpleAI-v0",
                        help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=5000,
                        help="Number of episodes to train for")
    parser.add_argument("--print_stats", type=bool, default=True)
    parser.add_argument("--run_id", type=int, default=0)

    return parser.parse_args(args)


# Entry point of the script
def main(args):
    # hyperparameters
    H = 50  # number of hidden layer neurons
    batch_size = 300  #
    learning_rate = 1e-4
    gamma = 0.99  # discount factor for reward
    decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
    mom_rate = 0.9
    td_step = 30  # initial td step
    gamma_power = [gamma ** i for i in range(td_step + 1)]
    shrink_step = True
    rmsprop = True
    resume = False  # resume from previous checkpoint?
    render = False

    # model initialization
    D = 50 * 50  # input dimensionality: 80x80 grid
    if resume:
        model, model_target = pickle.load(open('save.ac', 'rb'))
    else:
        model = {}
        model['W1_policy'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
        model['b1_policy'] = np.random.randn(H) / np.sqrt(4 * H)
        model['W2_policy'] = np.random.randn(H) / np.sqrt(H)
        model['b2_policy'] = 0.0
        model['W1_value'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
        model['b1_value'] = np.random.randn(H) / np.sqrt(4 * H)
        model['W2_value'] = np.random.randn(H) / np.sqrt(H)
        model['b2_value'] = 0.0
        model_target = copy.deepcopy(model)

    grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  # update buffers that add up gradients over a batch
    momentum = {k: np.zeros_like(v) for k, v in model.items()}
    rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]

    def prepro(observation, previous_observation):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        observation = observation[::4, ::4, 0]  # downsample by factor of 4
        observation[observation == 33] = 0  # erase background
        observation[(observation == 75) | (observation == 202) | (observation == 255)] = 1
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
        observation = observation.astype(np.float)
        # Show after substraction
        # plt.imshow(observation)
        # plt.colorbar()
        # plt.title("Observation after subtraction of previous image")
        # plt.show()
        # return I.astype(np.float).ravel()
        return observation.astype(np.float).ravel()


    def forward(x, modelType, model=model):
        h = np.dot(model['W1_' + modelType], x) + model['b1_' + modelType]
        h[h < 0] = 0  # ReLU nonlinearity
        out = np.dot(model['W2_' + modelType], h) + model['b2_' + modelType]
        if modelType == 'policy':
            out = sigmoid(out)
        return out, h

    def backward(eph, epx, epd, modelType):
        """ backward pass. (eph is array of intermediate hidden states) """
        db2 = sum(epd)[0]
        dW2 = np.dot(eph.T, epd).ravel()
        dh = np.outer(epd, model['W2_' + modelType])
        dh[eph <= 0] = 0  # backpro prelu
        db1 = sum(dh)
        dW1 = np.dot(dh.T, epx)
        return {'W1_' + modelType: dW1, 'W2_' + modelType: dW2, 'b1_' + modelType: db1, 'b2_' + modelType: db2}

    env = gym.make("WimblepongVisualSimpleAI-v0")
    num_episodes = args.train_episodes

    observation = env.reset()
    prev_x = None  # used in computing the difference frame
    xs, h_ps, h_vs, dlogps, vs, tvs, dvs = [], [], [], [], [], [], []
    running_reward = None
    reward_sum = 0
    round_number = 0
    ep = 0
    cumulative_rewards = []
    start = True
    frames = np.zeros((50, 50, 4))  # frame nr = 4
    while ep < num_episodes:

        # for ep in range(num_episodes):

        if render: env.render()

        # preprocess the observation, set input to network to be difference image
        cur_x = prepro(observation, start)  # observation, previous_observation, start,frames, frame_number=4
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        start = False
        x = cur_x
        # prev_x = cur_x

        # forward the policy network and sample an action from the returned probability
        aprob, h_p = forward(x, 'policy')
        action = 2 if np.random.uniform() < aprob else 3  # roll the dice!

        v, h_v = forward(x, 'value')
        tv, _ = forward(x, 'value', model_target)
        # record various intermediates (needed later for backprop)
        xs.append(x)  # observation
        h_ps.append(h_p)  # hidden state
        h_vs.append(h_v)
        vs.append(v)
        tvs.append(tv)
        y = 1 if action == 2 else 0  # a "fake label"
        dlogps.append(
            y - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward

        if reward != 0:
            round_number += 1
            if shrink_step and round_number % 10000 == 0:
                if td_step > 15:
                    td_step -= 1
            # calcutate td error
            dvs = [0] * len(vs)
            for i in range(len(vs)):
                if len(vs) - 1 - i < td_step:
                    dvs[i] = reward * (gamma_power[len(vs) - 1 - i]) - vs[i]
                else:
                    dvs[i] = gamma_power[td_step] * tvs[i + td_step] - vs[i]

            # stack together all inputs, hidden states, action gradients, and td for this episode
            epx = np.vstack(xs)
            eph_p = np.vstack(h_ps)
            eph_v = np.vstack(h_vs)
            epdlogp = np.vstack(dlogps)
            epv = np.vstack(dvs)
            xs, h_ps, h_vs, dlogps, vs, tvs, dvs = [], [], [], [], [], [], []  # reset array memory

            # discounted_epv = epv * np.vstack([gamma**i for i in range(len(epv))])
            epdlogp *= epv  # modulate the gradient with advantage (PG magic happens right here.)
            grad_p = backward(eph_p, epx, epdlogp, 'policy')
            grad_v = backward(eph_v, epx, epv, 'value')
            grad = dict(grad_p, **grad_v)

            for k in model: grad_buffer[k] += grad[k]  # accumulate grad over batch

            if round_number % batch_size == 0:
                for k, v in model.items():
                    g = grad_buffer[k]  # gradient
                    if rmsprop:
                        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                        momentum[k] = mom_rate * momentum[k] + learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                    else:
                        momentum[k] = mom_rate * momentum[k] + learning_rate * g
                    model[k] += momentum[k]
                    grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

                    if 'value' in k:
                        model_target[k] = mom_rate * model_target[k] + (1 - mom_rate) * model[k]

            if round_number % 3000 == 0:
                pickle.dump((model, model_target), open('save.ac', 'wb'))
        # boring book-keeping
        if done:
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            cumulative_rewards.append(reward_sum)
            print('Episode %d reward total was %.2f running mean: %.6f' % (ep, reward_sum, running_reward))
            ep += 1
            reward_sum = 0
            observation = env.reset()  # reset env
            prev_x = None
            if ep == 10000 or ep == 20000 or ep == 35000 or ep == 55000 or ep == 75000 or ep == 90000:
                plot_rewards(cumulative_rewards)
    plot_rewards(cumulative_rewards, 0)


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

    # plt.savefig('train_plot_%.png')
    # plt.savefig('train_plot_' + index + '.png')
    plt.savefig('train_plot_train_ac_{}.png'.format(index), format='png')

    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)
