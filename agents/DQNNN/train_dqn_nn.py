import matplotlib.pyplot as plt
import argparse
import sys
import gym
from random import randint
import pickle
import gym
import numpy as np
import argparse

import torch

import wimblepong
from PIL import Image
from agents.DQNNN.dqn_nn_agent import Agent, Network
from torch.utils.tensorboard import SummaryWriter


# parser = argparse.ArgumentParser()
# parser.add_argument("--headless", action="store_true", help="Run in headless mode")
# parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
# parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
# args = parser.parse_args()

def parse_args(args=sys.argv[1:]):
    # TODO [nice-to-have] lag continue training taking a file of weights already pretrained
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Directory to agent 1 to be tested.")
    parser.add_argument("--env", type=str, default="WimblepongVisualSimpleAI-v0",
                        help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=30000,
                        help="Number of episodes to train for")
    parser.add_argument("--print_stats", type=bool, default=True)
    parser.add_argument("--run_id", type=int, default=0)
    return parser.parse_args(args)


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
    return observation, subtract_next


def main(args):
    # Make the environment
    env = gym.make("WimblepongVisualSimpleAI-v0")
    # Number of episodes/games to play

    episodes = 100000
    target_update = 250

    # Define the player IDs for both SimpleAI agents
    player_id = 1
    agent = Agent(player_id)

    writer = SummaryWriter()

    agent.random_start_iter = 3
    final_epsilon = 0.05
    glie_a = 5555
    initial_epsilon = 1.0
    iteraction = 0
    num_episodes = args.train_episodes
    exploration_eps = 250000
    update_frequency = 4
    agent.epsilon = initial_epsilon
    cumulative_rewards = []
    replays = 0
    for ep in range(num_episodes):
        # Initialize the environment and state
        state = env.reset()

        state, previous_state = preprocess(state)

        done = False
        eps = glie_a / (glie_a + ep)
        cum_reward = 0
        timesteps = 0
        while not done:
            # Select and perform an action
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            # next_state, previous_state = preprocess(next_state, previous_state)
            cum_reward += reward
            timesteps += 1

            # Update the DQN
            # agent.store_transition(state, action, next_state, reward, done)
            if ep % update_frequency == 0:
                iteraction += 1
                replays += 1
                agent.update_network()

            if replays % update_frequency == 0:
                agent.update_target_network()

            agent.epsilon = max(final_epsilon, initial_epsilon - iteraction / exploration_eps)
            # Move to the next state
            # state = next_state

        print("Episode:", ep, "Reward: ", cum_reward, " ", "epsilon:", eps, "timesteps:", timesteps)
        cumulative_rewards.append(cum_reward)
        writer.add_scalar('Training ' + "PongEnv", cum_reward, ep)

        if ep == 10000 or ep == 20000 or ep == 15000:
            plot_rewards(cumulative_rewards, agent)

        # Save the policy
        # Uncomment for Task 4
        # if ep % 1000 == 0:
        #     torch.save(agent.policy_net.state_dict(),
        #                "weights_%s_%d.mdl" % ("PongEnv", ep))

    plot_rewards(cumulative_rewards, agent)
    print('Complete')
    plt.ioff()
    plt.show()


def plot_rewards(rewards, agent):
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
    plt.savefig('train_plot.png')
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
    main(args=args)
