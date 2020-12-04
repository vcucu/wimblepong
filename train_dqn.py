import argparse
import sys
import gym
import time
import numpy as np
import wimblepong
import torch
import matplotlib.pyplot as plt



def parse_args(args=sys.argv[1:]):
    # TODO [nice-to-have] lag continue training taking a file of weights already pretrained
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Directory to agent 1 to be tested.")
    parser.add_argument("--env", type=str, default="WimblepongVisualSimpleAI-v0",
                        help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=100000,
                        help="Number of episodes to train for")
    parser.add_argument("--print_stats", type=bool, default=True)
    parser.add_argument("--run_id", type=int, default=0)
    return parser.parse_args(args)


def main(args):
    # Create a Gym environment
    env = gym.make(args.env)
    TARGET_UPDATE = 4
    glie_a = 6666
    num_episodes = args.train_episodes
    total_timesteps = 0
    sys.path.append(args.dir)
    from agents import DQN as model
    agent = model.DQNAgent()

    cumulative_rewards = []
    for ep in range(num_episodes):
        # Initialize the environment and state
        state = env.reset()
        done = False
        eps = glie_a / (glie_a + ep)
        cum_reward = 0
        timesteps = 0
        #start = time.time()
        while not done:
            timesteps += 1
            total_timesteps += 1
            # Select and perform an action
            action = agent.get_action(state, eps)
            next_state, reward, done, _ = env.step(action)
            cum_reward += reward
            # Update the DQN
            agent.store_transition(state, action, next_state, reward, done)
            #net_update_start = time.time()
            agent.update_network()
            #net_update_end = time.time()
            #print("This network update took", net_update_end - net_update_start)
            # Move to the next state
            state = next_state
        #end = time.time()
        #print("This episode took", end - start)

        print("Episode:", ep, "Reward: ", cum_reward, "epsilon:", eps, "timesteps:", timesteps)
        cumulative_rewards.append(cum_reward)

        # Update the target network, copying all weights and biases in DQN
        if ep % TARGET_UPDATE == 0:
            agent.update_target_network()

        # Save the policy
        if ep % 1000 == 0:
            torch.save(agent.policy_net.state_dict(),
                       "weights_%s_%d.mdl" % ("PongEnv", ep))

    plot_rewards(cumulative_rewards, agent)
    print('Complete, ran ', total_timesteps, 'timesteps in total')
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
