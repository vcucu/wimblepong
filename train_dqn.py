import argparse
import sys
import gym
import seaborn

import wimblepong
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def parse_args(args=sys.argv[1:]):
    # TODO [nice-to-have] lag continue training taking a file of weights already pretrained
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Directory to agent 1 to be tested.", default="agents/PG")
    parser.add_argument("--env", type=str, default="WimblepongVisualSimpleAI-v0",
                        help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=20000,
                        help="Number of episodes to train for")
    parser.add_argument("--print_stats", type=bool, default=True)
    parser.add_argument("--run_id", type=int, default=0)

    return parser.parse_args(args)


def preprocess(observation):
    observation = observation[::2, ::2, 0]  # downsample by factor of 2
    observation[observation == 144] = 0  # erase background (background type 1)
    observation[observation == 109] = 0  # erase background (background type 2)
    observation[observation != 0] = 1  # everything else (paddles, ball) just set to 1
    observation = observation.astype(np.float).ravel()
    return observation


def main(args):
    # Create a Gym environment
    env = gym.make(args.env)

    TARGET_UPDATE = 4
    glie_a = 100
    num_episodes = args.train_episodes
    hidden = 32
    gamma = 0.99
    replay_buffer_size = 5000
    batch_size = 100

    writer = SummaryWriter()
    n_actions = 3
    # TODO: change when we preprocess the observation
    state_space_dim = env.observation_space.shape

    sys.path.append(args.dir)
    from agents import DQN as model
    agent = model.DQNAgent(env_name = env, state_space = state_space_dim, n_actions = n_actions,
                           replay_buffer_size = replay_buffer_size, batch_size = batch_size,
                           hidden_size = hidden, gamma = gamma)

    cumulative_rewards = []
    for ep in range(num_episodes):

        # Initialize the environment and state
        state = env.reset()
        state = preprocess(state)
        done = False
        eps = glie_a / (glie_a + ep)
        cum_reward = 0
        while not done:
            # Select and perform an action

            action = agent.get_action(state, eps)
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess(next_state)
            cum_reward += reward

            # Task 1: Update the Q-values
            # agent.single_update(state, action, next_state, reward, done)

            # Task 2:  Store transition and batch-update Q-values
            #agent.store_transition(state, action, next_state, reward, done)
            #agent.update_estimator()

            # Task 4: Update the DQN
            agent.store_transition(state, action, next_state, reward, done)
            agent.update_network()

            # Move to the next state
            state = next_state

        print(ep, cum_reward)
        cumulative_rewards.append(cum_reward)
        writer.add_scalar('Training ' + "PongEnv", cum_reward, ep)

        # Update the target network, copying all weights and biases in DQN
        # Uncomment for Task 4
        if ep % TARGET_UPDATE == 0:
             agent.update_target_network()

        # Save the policy
        # Uncomment for Task 4
        if ep % 1000 == 0:
             torch.save(agent.policy_net.state_dict(),
                       "weights_%s_%d.mdl" % ("PongEnv", ep))

    plot_rewards(cumulative_rewards, agent)
    print('Complete')
    plt.ioff()
    plt.show()





def plot_rewards(rewards,agent):
    # plot the policy
    discr = 64
    x_min, x_max = -2.4, 2.4
    th_min, th_max = -0.3, 0.3
    num_of_actions = 2

    # Create discretization grid
    x_grid = np.linspace(x_min, x_max, discr)
    th_grid = np.linspace(th_min, th_max, discr)
    q_grid = np.zeros((discr, discr))

    # Construct policy
    for x in x_grid:
        for th in th_grid:
            x_d, th_d = discretize(x, th, x_grid, th_grid)
            state = np.array([x, 0, th, 0])
            action = agent.get_action(state)
            q_grid[th_d][x_d] = action

    seaborn.heatmap(q_grid)
    plt.title("Policy visualized over x-location(x-axis) and angle theta(y-axis)")
    plt.show()

def find_nearest(array, value):
    return np.argmin(abs(array - value))

def discretize(x, th, x_grid, th_grid):
    x_ = find_nearest(x_grid, x)
    th_ = find_nearest(th_grid, th)
    return x_, th_


# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args = args)
