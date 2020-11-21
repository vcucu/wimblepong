import argparse
import sys
import gym
import wimblepong


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Directory to agent 1 to be tested.")
    parser.add_argument("--env", type=str, default="WimblepongSimpleAI-v0",
                        help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=500,
                        help="Number of episodes to train for")
    return parser.parse_args(args)


def main(args):
    # np.random.seed(123) #is this useful?

    env = gym.make(args.env)
    env.seed(321)

    episodes = args.train_episodes
    actions_num = 3
    sys.path.append(args.dir)
    from agents import Qlearning as q
    agent = q.Agent(env)  # This is not gonna work
    agent.train(episodes, env, actions_num)


# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)
