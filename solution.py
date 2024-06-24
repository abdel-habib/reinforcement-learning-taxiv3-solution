import argparse
import gymnasium as gym
import random
import os

# importing functions from utils
from helpers.loggers import logger
import helpers.utils as utils
import helpers.agents as agents

if __name__ == "__main__":
    # optional arguments from the command line 
    parser = argparse.ArgumentParser()

    parser.add_argument('--algorithm', type=str, default='qlearning', help='Algorithm to solve the task. Has to be one of ["qlearning", "random", "sarsa"].')
    parser.add_argument('--n_episods', type=int, default=10000, help='Number of episodes to use for training. Default value is `10000`.')
    parser.add_argument('--force_visualize', type=bool, default=False, help='visualizing is disabled for random agent solution, as it will fill the command line. \
        To visualze the test simulation of the random agent, set this flag to `True`.')
    parser.add_argument('--export_metrics', action='store_true', help='Set this flag to `True` when it is necessary to export performance metrics, such as \
        `penality per episode` or `epochs per episode`. This automatically creates an outputs folder, and saves the filename based on the timestamp.')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory to export the figures. Default is `.\outputs`.')


    # parse the arguments
    args = parser.parse_args()

    # assert args values and create necessary folders if needed
    assert args.algorithm in ["random", "qlearning", "sarsa"], \
        'Wronge input values for the algorithm to solve the challenge! Pass one of ["qlearning", "random", "sarsa"] in the argument `--algorithm`.'
    
    if args.export_metrics:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    # create an environment
    env = gym.make("Taxi-v3", render_mode="ansi")

    # [optional] seed random generators and the environment to replicate results
    # utils.seed(42, env)

    # define values to use in the experiment, extract them from the parsed args if necessary
    n_episodes = agents.Value(data=args.n_episods, label='n_Episodes')

    # choose which agent to run the file with, based on user input
    if args.algorithm == "random":
        agent = agents.RandomAgent(env)
    elif args.algorithm == "qlearning":
        agent = agents.QLearningAgent(env, epsilon = 0.1, n_episodes = n_episodes.data)
    elif args.algorithm == "sarsa":
        agent = agents.SARSAAgent(env, epsilon = 0.1, n_episodes = n_episodes.data)

    # For plotting metrics
    epochs_per_episode = []
    penalties_per_episode = []

    # update the training function for all of the agents to return both lists above



