import argparse
import gymnasium as gym
import time
import pandas as pd
import numpy as np

# importing functions from utils
from helpers.loggers import logger
import helpers.utils as utils
import helpers.agents as agents

if __name__ == "__main__":
    # optional arguments from the command line 
    parser = argparse.ArgumentParser()

    # define fixed values or parameters for all models
    n_episodes = agents.Value(data=10000, label='n_Episodes')
    env = gym.make("Taxi-v3", render_mode="ansi")
    utils.seed(42, env)

    # here we load the agents we want to evaluate, we don't load the random agent as we know that randomness doesn't converge to a solution
    ql_agent = agents.QLearningAgent(env, epsilon = 0.1, n_episodes = n_episodes.data)
    sarsa_agent = agents.SARSAAgent(env, epsilon = 0.1, n_episodes = n_episodes.data)

    # store the results in a dict
    results = {}
    results['ql_agent'], results['sarsa_agent'] = {}, {}

    # start training
    # we track the training time, although they are trained for same number of episodes, it is not definite that both approaches will need the same 
    # amout of epochs for every epoch
    logger.info(f"Training QLearningAgent agent for {n_episodes.data} episods...")
    start = time.time()
    _, _= ql_agent.train()
    end = time.time()
    results['ql_agent']['training_time'] = end-start

    logger.info(f"Training SARSAAgent agent for {n_episodes.data} episods...")
    start = time.time()
    _, _ = sarsa_agent.train()
    end = time.time()
    results['sarsa_agent']['training_time'] = end-start

    ql_agent_test_steps, ql_agent_total_rewards = [], []
    sarsa_agent_test_steps, sarsa_agent_total_rewards = [], []

    logger.info(f"Running test on 100 samples...")
    for i in range(100):
        # run 100 random tests and average the results
        test_step, total_rewards = ql_agent.test(visualize = False)
        ql_agent_test_steps.append(test_step)
        ql_agent_total_rewards.append(total_rewards)

        test_step, total_rewards = sarsa_agent.test(visualize = False)
        sarsa_agent_test_steps.append(test_step)
        sarsa_agent_total_rewards.append(total_rewards)

    results['ql_agent']['average training epochs (n=100)'], results['ql_agent']['average total rewards'] = np.mean(ql_agent_test_steps), np.mean(ql_agent_total_rewards)
    results['sarsa_agent']['average training epochs (n=100)'], results['sarsa_agent']['average total rewards'] = np.mean(sarsa_agent_test_steps), np.mean(sarsa_agent_total_rewards)
    print(pd.DataFrame.from_dict(results, orient='index'))

    # each agent has a q-table that we will use for testing to see how efficient is it able to obtain a solution
