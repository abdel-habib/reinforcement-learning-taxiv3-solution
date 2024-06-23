import random
import numpy as np
import matplotlib.pyplot as plt

def show_state(step, env, obs, reward):
    '''Helper function to render the state of the environment and print it to the console.

    Args:
        step (int): The current step in the environment.
        env (gym.Env): The environment to render.
        obs (int): The observation of the environment.
        reward (int): The reward from the environment.

    Returns:
        None
    '''
    ansi_state = env.render()
    array_state = list(env.unwrapped.decode(obs))
    print(f"Step {step}: {array_state}, Reward: {reward}")
    print(ansi_state)

def plot_learning(x_values, y_values, legend_title='Penalities per Episode', ylabel='Penalities'):
    '''Helper function to plot the learning curve of an agent.'''
    
    plt.figure(figsize=(12, 4))
    
    # Plotting the data
    plt.plot(x_values, y_values, label=legend_title)  # Use legend_title as the label for the plot

    # Adding labels and title
    plt.xlabel('Number of Episodes')
    plt.ylabel(ylabel)
    plt.title('Learning Curve')  # Adding a title to the plot

    # Display the legend with the specified title
    plt.legend([legend_title])  # Pass legend_title as a list to plt.legend()

    # Display the plot
    plt.show()


def seed(seed, env):
    '''Helper function to seed the environment and all random number generators.

    Args:
        seed (int): The seed value to use for seeding the environment and all random number generators.

    Returns:
        None
    '''
    # global seeding
    random.seed(seed)
    np.random.seed(seed)

    # seeding the environment
    env.unwrapped.s = seed
    