import argparse
import sys
import gym
import wimblepong
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


def parse_args(args=sys.argv[1:]):
    # TODO [nice-to-have] lag continue training taking a file of weights already pretrained
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Directory to agent 1 to be tested.")
    parser.add_argument("--env", type=str, default="WimblepongVisualSimpleAI-v0",
                        help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=500,
                        help="Number of episodes to train for")
    parser.add_argument("--print_stats", type=bool, default=True)
    parser.add_argument("--run_id", type=int, default=0)

    return parser.parse_args(args)


def main(args):
    # Create a Gym environment
    env = gym.make(args.env)

    action_space_dim = 1
    # TODO: change when we preprocess the observation
    observation_space_dim = env.observation_space.shape
    sys.path.append(args.dir)
    from agents import DQN as model

    # Instantiate agent and its policy
    policy = model.Policy(observation_space_dim, action_space_dim)
    agent = model.Agent(env=env, policy=policy)

    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []

    # Run actual training
    for episode_number in range(args.train_episodes):
        reward_sum, timesteps = 0, 0
        done = False
        # Reset the environment and observe the initial state
        observation = env.reset()

        # Loop until the episode is over
        while not done:
            # Get action from the agent
            action, action_probabilities = agent.get_action(observation)
            previous_observation = observation

            # Perform the action on the environment, get new state and reward
            observation, reward, done, info = env.step(action.detach().numpy())

            # Store action's outcome (so that the agent can improve its policy)
            agent.store_outcome(previous_observation, action_probabilities, action, reward)

            # Store total episode reward
            reward_sum += reward
            timesteps += 1

        print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
              .format(episode_number, reward_sum, timesteps))

        # Bookkeeping (mainly for generating plots)
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        if episode_number > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        # Let the agent do its magic (update the policy)
        agent.episode_finished(episode_number)

    # Training is finished - plot rewards
    if args.print_stats:
        plt.plot(reward_history)
        plt.plot(average_reward_history)
        plt.legend(["Reward", "100-episode average"])
        plt.title("Reward history")
        plt.show()
        print("Training finished.")
        data = pd.DataFrame({"episode": np.arange(len(reward_history)),
                             "train_run_id": [args.train_run_id] * len(reward_history),
                             "algorithm": ["PG no baseline"] * len(reward_history),
                             "reward": reward_history})
    torch.save(agent.policy.state_dict(), "model_%s_%d.mdl" % ("PongEnv", args.train_run_id))
    return data


# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)
