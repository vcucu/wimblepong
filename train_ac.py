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
    parser.add_argument("--dir", type=str, help="Directory to agent 1 to be tested.", default="agents/PG")
    parser.add_argument("--env", type=str, default="WimblepongVisualSimpleAI-v0",
                        help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=5000,
                        help="Number of episodes to train for")
    parser.add_argument("--print_stats", type=bool, default=True)
    parser.add_argument("--run_id", type=int, default=0)

    return parser.parse_args(args)

# Policy training function
# def train(policy,agent, print_things=True, train_run_id=0, train_episodes=5000):
#     # Create a Gym environment
#     env = gym.make("WimblepongVisualSimpleAI-v0")
#
#     # Arrays to keep track of rewards
#     reward_history, timestep_history = [], []
#     average_reward_history = []
#
#     # Run actual training
#     for episode_number in range(train_episodes):
#         reward_sum, timesteps = 0, 0
#         done = False
#         # Reset the environment and observe the initial state
#         observation = env.reset()
#
#         # Loop until the episode is over
#         while not done:
#             # Get action from the agent
#             action, action_probabilities = agent.get_action(observation)
#             previous_observation = observation
#
#             # Perform the action on the environment, get new state and reward
#             observation, reward, done, info = env.step(action.detach().numpy())
#
#             # Store action's outcome (so that the agent can improve its policy)
#             agent.store_outcome(previous_observation, action_probabilities, action, reward)
#
#             # Store total episode reward
#             reward_sum += reward
#             timesteps += 1
#
#         if print_things:
#             print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
#                   .format(episode_number, reward_sum, timesteps))
#
#         # Bookkeeping (mainly for generating plots)
#         reward_history.append(reward_sum)
#         timestep_history.append(timesteps)
#         if episode_number > 100:
#             avg = np.mean(reward_history[-100:])
#         else:
#             avg = np.mean(reward_history)
#         average_reward_history.append(avg)
#
#         # Let the agent do its magic (update the policy)
#         agent.episode_finished(episode_number) # TODO: Update at end of each episode -- DONE
#
#     # Training is finished - plot rewards
#     if print_things:
#         plt.plot(reward_history)
#         plt.plot(average_reward_history)
#         plt.legend(["Reward", "100-episode average"])
#         plt.title("Reward history")
#         plt.show()
#         print("Training finished.")
#     data = pd.DataFrame({"episode": np.arange(len(reward_history)),
#                          "train_run_id": [train_run_id]*len(reward_history),
#                          # TODO: Change algorithm name for plots, if you want -- DONE
#                          "REINFORCE without baseline": ["PG"]*len(reward_history),
#                          # "REINFORCE constant baseline": ["PG"]*len(reward_history),
#                          # "REINFORCE normalized discounted rewards and unit variance": ["PG"]*len(reward_history),
#                          "reward": reward_history})
#     torch.save(agent.policy.state_dict(), "model_%s_%d.mdl" % (env_name, train_run_id))
#     return data
def train(env, agent, print_things=True, train_run_id=0, train_episodes=5000):

    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []

    # Run actual training
    for episode_number in range(train_episodes):
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

        if print_things:
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
        agent.episode_finished(episode_number) # TODO: Update at end of each episode -- DONE

    # Training is finished - plot rewards
    if print_things:
        plt.plot(reward_history)
        plt.plot(average_reward_history)
        plt.legend(["Reward", "100-episode average"])
        plt.title("Reward history")
        plt.show()
        print("Training finished.")
    data = pd.DataFrame({"episode": np.arange(len(reward_history)),
                         "train_run_id": [train_run_id]*len(reward_history),
                         # TODO: Change algorithm name for plots, if you want -- DONE
                         "REINFORCE without baseline": ["PG"]*len(reward_history),
                         # "REINFORCE constant baseline": ["PG"]*len(reward_history),
                         # "REINFORCE normalized discounted rewards and unit variance": ["PG"]*len(reward_history),
                         "reward": reward_history})
    torch.save(agent.policy.state_dict(), "model_%s_%d.mdl" % (env_name, train_run_id))
    return data


# Function to test a trained policy
# def test(env_name, episodes, params, render):
#     # Create a Gym environment
#     env = gym.make(env_name)
#
#     # Get dimensionalities of actions and observations
#     action_space_dim = env.action_space.shape[-1]
#     observation_space_dim = env.observation_space.shape[-1]
#
#     # Instantiate agent and its policy
#     policy = Policy(observation_space_dim, action_space_dim)
#     policy.load_state_dict(params)
#     agent = Agent(policy)
#
#     test_reward, test_len = 0, 0
#     for ep in range(episodes):
#         done = False
#         observation = env.reset()
#         while not done:
#             # Similar to the training loop above -
#             # get the action, act on the environment, save total reward
#             # (evaluation=True makes the agent always return what it thinks to be
#             # the best action - there is no exploration at this point)
#             action, _ = agent.get_action(observation, evaluation=True)
#             observation, reward, done, info = env.step(action.detach().cpu().numpy())
#
#             if render:
#                 env.render()
#             test_reward += reward
#             test_len += 1
#     print("Average test reward:", test_reward/episodes, "episode length:", test_len/episodes)

def main(args):
    # Create a Gym environment
    env = gym.make(args.env)

    action_space_dim = 1
    # TODO: change when we preprocess the observation
    sys.path.append(args.dir)
    from agents import DQN as model
    from agents.AC import ac_agent as model

    # hyperparameters for REINFORCE
    batch_size = 10  # every how many episodes to do a param update?
    learning_rate = 1e-3
    gamma = 0.99  # discount factor for reward
    decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
    resume = False  # resume from previous checkpoint?
    render = False

    # hyperparameters for ActorCritic

    mom_rate = 0.9
    td_step = 30  # initial td step
    gamma_power = [gamma ** i for i in range(td_step + 1)]
    shrink_step = True
    rmsprop = True
    state_space_dim = 50*50
    hidden = 32
    n_actions = 3

    # Instantiate agent and its policy
    policy = model.Policy(state_space_dim, n_actions)
    agent = model.Agent(policy=policy )
    train(env,agent=agent)#, batch_size, learning_rate, gamma, decay_rate)#, mom_rate, td_step, gamma_power, shrink_step,  rmsprop, render)


# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)
