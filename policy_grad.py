import argparse
import datetime
import os
import random
import time
import argparse
import sys
import wimblepong
import gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import PIL.Image as Image


def preprocess(image):
    """ Pre-process 210x160x3 uint8 frame into 6400 (80x80) 1D float vector. """

    image = torch.Tensor(image)

    image = image[::4, ::4, 0]  # downsample by factor of 2
    image[image == 33] = 0  # erase background
    image[(image == 75) | (image == 202) | (image == 255)] = 1
    # plt.imshow(image)
    # plt.colorbar()
    # plt.title("Observation after downsampling and color encoding")
    # plt.show()

    # image = Image.fromarray(image)
    # image = image.convert("L")
    # image = image.resize((25, 25), Image.NEAREST)
    # image = tf.image.rgb_to_grayscale(image, name=None)
    return image.flatten().float()


def calc_discounted_future_rewards(rewards, discount_factor):
    r"""
    Calculate the discounted future reward at each timestep.
    discounted_future_reward[t] = \sum_{k=1} discount_factor^k * reward[t+k]
    """

    discounted_future_rewards = torch.empty(len(rewards))

    # Compute discounted_future_reward for each timestep by iterating backwards
    # from end of episode to beginning
    discounted_future_reward = 0
    for t in range(len(rewards) - 1, -1, -1):
        # If rewards[t] != 0, we are at game boundary (win or loss) so we
        # reset discounted_future_reward to 0 (this is pong specific!)
        if rewards[t] != 0:
            discounted_future_reward = 0

        discounted_future_reward = rewards[t] + discount_factor * discounted_future_reward
        discounted_future_rewards[t] = discounted_future_reward

    return discounted_future_rewards


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


def run_episode(model, env, discount_factor, render=False):
    UP = 2
    DOWN = 3

    observation = env.reset()
    prev_x = preprocess(observation)

    action_chosen_log_probs = []
    rewards = []

    done = False
    timestep = 0

    while not done:
        if render:
            # Render game window at 30fps
            time.sleep(1 / 30)
            env.render()

        # Preprocess the observation, set input to network to be difference
        # image between frames
        cur_x = preprocess(observation)
        x = cur_x - prev_x
        prev_x = cur_x

        # Run the policy network and sample action from the returned probability
        prob_up = model(x)
        action = UP if random.random() < prob_up else DOWN  # roll the dice!

        # Calculate the probability of sampling the action that was chosen
        action_chosen_prob = prob_up if action == UP else (1 - prob_up)
        action_chosen_log_probs.append(torch.log(action_chosen_prob))

        # Step the environment, get new measurements, and updated discounted_reward
        observation, reward, done, info = env.step(action)
        rewards.append(torch.Tensor([reward]))
        timestep += 1

    # Concat lists of log probs and rewards into 1-D tensors
    action_chosen_log_probs = torch.cat(action_chosen_log_probs)
    rewards = torch.cat(rewards)

    # Calculate the discounted future reward at each timestep
    discounted_future_rewards = calc_discounted_future_rewards(rewards, discount_factor)

    # Standardize the rewards to have mean 0, std. deviation 1 (helps control the gradient estimator variance).
    # It encourages roughly half of the actions to be rewarded and half to be discouraged, which
    # is helpful especially in beginning when positive reward signals are rare.
    discounted_future_rewards = (discounted_future_rewards - discounted_future_rewards.mean()) \
                                / discounted_future_rewards.std()

    # PG magic happens right here, multiplying action_chosen_log_probs by future reward.
    # Negate since the optimizer does gradient descent (instead of gradient ascent)
    loss = -(discounted_future_rewards * action_chosen_log_probs).sum()

    return loss, rewards.sum()


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


def main():
    # By default, doesn't render game screen, but can invoke with `--render` flag on CLI

    args = parse_args()

    # Hyperparameters
    input_size = 50 * 50  # input dimensionality: 80x80 grid
    hidden_size = 200  # number of hidden layer neurons
    learning_rate = 7e-4
    discount_factor = 0.99  # discount factor for reward

    batch_size = 4
    save_every_batches = 5

    # Create policy network
    model = PolicyNetwork(input_size, hidden_size)

    # Load model weights and metadata from checkpoint if exists
    if os.path.exists('checkpoint2.pth'):
        print('Loading from checkpoint...')
        save_dict = torch.load('checkpoint2.pth')

        model.load_state_dict(save_dict['model_weights'])
        start_time = save_dict['start_time']
        last_batch = save_dict['last_batch']
    else:
        start_time = datetime.datetime.now().strftime("%H.%M.%S-%m.%d.%Y")
        last_batch = -1

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Set up tensorboard logging
    # tf_writer = tf.summary.create_file_writer(
    #     os.path.join('tensorboard_logs', start_time))
    # tf_writer.set_as_default()

    # Create pong environment (PongDeterministic versions run faster)
    env = gym.make("WimblepongVisualSimpleAI-v0")

    # Pick up at the batch number we left off at to make tensorboard plots nicer
    batch = last_batch + 1
    ep = 0
    cumulative_rewards = []
    while ep < args.train_episodes:

        mean_batch_loss = 0
        mean_batch_reward = 0
        for batch_episode in range(batch_size):
            # Run one episode
            loss, episode_reward = run_episode(model, env, discount_factor)
            ep += 1
            mean_batch_loss += loss / batch_size
            mean_batch_reward += episode_reward / batch_size
            cumulative_rewards.append(episode_reward)

            # Boring book-keeping
            print(f'Episode {ep} reward total was {episode_reward}')

        # Backprop after `batch_size` episodes
        optimizer.zero_grad()
        mean_batch_loss.backward()
        optimizer.step()

        # Batch metrics and tensorboard logging
        # print(f'Batch: {batch}, mean loss: {mean_batch_loss:.2f}, '
        #       f'mean reward: {mean_batch_reward:.2f}')
        # # tf.summary.scalar('mean loss', mean_batch_loss.detach().item(), step=batch)
        # tf.summary.scalar('mean reward', mean_batch_reward.detach().item(), step=batch)

        if ep == 10000 or ep == 20000 or ep == 35000 or ep == 55000 or ep == 75000 or ep == 90000:
            plot_rewards(cumulative_rewards, ep)

        if batch % save_every_batches == 0:
            print('Saving checkpoint...')
            save_dict = {
                'model_weights': model.state_dict(),
                'start_time': start_time,
                'last_batch': batch
            }
            torch.save(save_dict, 'checkpoint2.pth')
        if ep == 10000 or ep == 20000 or ep == 35000 or ep == 55000 or ep == 75000 or ep == 90000:
            torch.save(save_dict,
                       "weights_%s_%d.mdl" % ("PongEnv_PG", ep))

        batch += 1
    return cumulative_rewards


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

    # plt.savefig('train_plott.png')
    plt.savefig('train_plot_{}.png'.format(index), format='png')

    plt.show()


if __name__ == '__main__':
    rev = main()
    plot_rewards(rev, 0)
