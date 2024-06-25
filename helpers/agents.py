import numpy as np
from .utils import show_state
from tqdm import tqdm

class Value():
    '''A class representing a value in a reinforcement learning environment. This class is
    used to represent the value of a state or action in an environment.
    
    Args:
        data (int/float/any): The value of the state or action.
        _children (tuple): A tuple of child values.
        _op (str): The operation performed to get the value.
        label (str): A label for the value.'''
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label.title() if label != '' and len(label.split()) == 1 else 'Value'
        
    def __repr__(self):
        return f"{self.label}(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        return out
    
    def __radd__(self, other): # other + self
        return self + other

    # def __neg__(self): # -self
    #     return self * -1
    
    def __sub__(self, other): # self - other
        other = other if isinstance(other, Value) else Value(other)
        return Value(self + (-other.data), (self, other), '-')


class RandomAgent():
    '''An agent that acts randomly by random sampling from the environment's action space.
    
    Args:
        env (gymnasium.Env): The environment to interact with.
        n_episodes (int): The number of episodes to train the agent for.
        '''

    def __init__(self, env, n_episodes=1000):
        self.env = env
        self.n_episodes = n_episodes

    def get_action(self, state=None):
        '''Get a random action from the environment's action space. 
        The state is not necessary in this function, it is only written to standerdise the implementation.
        
        Returns:
            action (int): A random action.
            
        '''
        return self.env.action_space.sample()
    
    def train(self):
        '''Train the agent in the environment.'''
        # to keep track of the learning
        epochs_per_episode = []
        penalties_per_episode = []

        for episode in tqdm(range(self.n_episodes)):
            # defining and keeping track of values
            penalties = Value(data=0, label='penalties')
            total_rewards = Value(data=0, label='penalties')    
            epochs = Value(data=0, label='epochs')
            
            # both terminated, truncated are returned on every step
            terminated, truncated = False, False
            
            # reset the environment
            obs, info = self.env.reset()

            # training loop
            while not terminated or not truncated:
                # get the action value
                action = self.get_action(obs)

                # take a step towards the solution
                obs, reward, terminated, truncated, info = self.env.step(action)
            
                # keep track of values
                total_rewards += reward
                if reward == -10:
                    penalties += reward
                
                epochs += 1

            epochs_per_episode.append(epochs.data)
            penalties_per_episode.append(penalties.data)

        return epochs_per_episode, penalties_per_episode

    
    def test(self, visualize = True):
        '''Test the agent in the environment.'''

        obs, info  = self.env.reset()

        if visualize:
            show_state(0, self.env, obs, 0)
        
        total_rewards = 0

        terminated, truncated = False, False
        test_step = 0

        while not terminated and not truncated:
            action = self.get_action()

            new_obs, reward, terminated, truncated, _ = self.env.step(action)
            obs = new_obs
            test_step += 1

            total_rewards += reward

            if visualize:
                show_state(test_step, self.env, obs, reward)

        return test_step, total_rewards

    
class QLearningAgent():
    '''An agent that learns the optimal policy using the Q-learning algorithm.
    
    Args:
        env (gymnasium.Env): The environment to interact with.
        alpha (float): The learning rate.
        gamma (float): The discount factor.
        epsilon (float): The exploration rate.
        n_episodes (int): The number of episodes to train the agent for.'''
    
    def __init__(self, env, alpha=0.1, gamma=0.90, epsilon=0.1, n_episodes=1000):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_episodes = n_episodes
        self.q_table = None

    def get_action(self, state):
        '''Get the action to take in a given state.
        
        Args:
            state (int): The current state of the environment.
        
        Returns:
            action (int): The action to take.'''
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()       # Exploration
        else:
            return np.argmax(self.q_table[state])    # Exploitation
        
    def update_q_table(self, state, action, reward, next_state):
        '''Update the Q-table using the Q-learning algorithm.
        
        Args:
            state (int): The current state of the environment.
            action (int): The action taken in the current state.
            reward (int): The reward received from the environment.
            next_state (int): The next state of the environment.'''
        
        self.q_table[state, action] = self.q_table[state, action] + self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state, action])

    def train(self):
        '''Train the agent using the Q-learning algorithm.'''
        # initialize the q_table, important to re-init every time we train
        self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))

        # to keep track of the learning        
        epochs_per_episode = []
        penalties_per_episode = []

        for episode in tqdm(range(self.n_episodes)):
            # defining and keeping track of values
            penalties = Value(data=0, label='penalties')
            total_rewards = Value(data=0, label='penalties')    
            epochs = Value(data=0, label='epochs')
            
            # both terminated, truncated are returned on every step
            terminated, truncated = False, False
            
            # reset the environment
            obs, info = self.env.reset()

            # training loop
            while not ((terminated) or (truncated)):
                # get the action value
                action = self.get_action(obs)

                # take a step towards the solution
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                # update the agent parameters
                self.update_q_table(obs, action, reward, next_obs)
            
                # keep track of values
                total_rewards += reward
                if reward == -10:
                    penalties += reward #1
                
                epochs += 1
                obs = next_obs # update the next state
            
            epochs_per_episode.append(epochs.data)
            penalties_per_episode.append(penalties.data)

        return epochs_per_episode, penalties_per_episode

    
    def test(self, visualize = True):
        '''Test the agent in the environment.'''
        
        obs, info  = self.env.reset()

        if visualize:
            show_state(0, self.env, obs, 0)

        terminated, truncated = False, False
        test_step = 0

        total_rewards = 0

        while not terminated and not truncated:
            action = np.argmax(self.q_table[obs])
            new_obs, reward, terminated, truncated, _ = self.env.step(action)
            obs = new_obs
            test_step += 1

            if visualize:
                show_state(test_step, self.env, obs, reward)

            total_rewards += reward

        return test_step, total_rewards



class SARSAAgent():
    '''An agent that learns the optimal policy using the SARSA algorithm.
    
    Args:
        env (gymnasium.Env): The environment to interact with.
        alpha (float): The learning rate.
        gamma (float): The discount factor.
        epsilon (float): The exploration rate.
        n_episodes (int): The number of episodes to train the agent for.'''
    
    def __init__(self, env, alpha=0.1, gamma=0.90, epsilon=0.1, n_episodes=1000):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_episodes = n_episodes
        self.q_table = None

    def get_action(self, state):
        '''Get the action to take in a given state.
        
        Args:
            state (int): The current state of the environment.
        
        Returns:
            action (int): The action to take.'''
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()       # Exploration
        else:
            return np.argmax(self.q_table[state])       # Exploitation
        
    def update_q_table(self, state, action, reward, next_state, next_action):
        '''Update the Q-table using the SARSA algorithm.
        
        Args:
            state (int): The current state of the environment.
            action (int): The action taken in the current state.
            reward (int): The reward received from the environment.
            next_state (int): The next state of the environment.
            next_action (int): The action taken in the next state.'''
        
        self.q_table[state, action] += self.alpha * (
            reward + self.gamma * self.q_table[next_state, next_action] - self.q_table[state, action])

    def train(self):
        '''Train the agent using the SARSA algorithm.'''
        # initialize the q_table, important to re-init every time we train
        self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))

        # to keep track of the learning
        epochs_per_episode = []
        penalties_per_episode = []

        for episode in tqdm(range(self.n_episodes)):
            # defining and keeping track of values
            penalties = Value(data=0, label='penalties')
            total_rewards = Value(data=0, label='penalties')    
            epochs = Value(data=0, label='epochs')
            
            # both terminated, truncated are returned on every step
            terminated, truncated = False, False
            
            # reset the environment
            obs, info = self.env.reset() # obs, info

            # training loop
            while not ((terminated) or (truncated)):
                # get the action value
                action = self.get_action(obs)

                # take a step towards the solution
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                next_action = self.get_action(next_obs)

                # update the agent parameters
                self.update_q_table(obs, action, reward, next_obs, next_action)
            
                # keep track of values
                total_rewards += reward
                if reward == -10:
                    penalties += reward #1
                
                epochs += 1
                obs, action = next_obs, next_action # update the next state
            
            epochs_per_episode.append(epochs.data)
            penalties_per_episode.append(penalties.data)

        return epochs_per_episode, penalties_per_episode

    def test(self, visualize = True):
        '''Test the agent in the environment.'''
        obs, info = self.env.reset()

        if visualize:
            show_state(0, self.env, obs, 0)

        terminated, truncated = False, False
        test_step = 0

        total_rewards = 0

        while not terminated and not truncated:
            action = np.argmax(self.q_table[obs])
            new_obs, reward, terminated, truncated, info = self.env.step(action)
            obs = new_obs
            test_step += 1

            if visualize:
                show_state(test_step, self.env, obs, reward)

            total_rewards += reward

        return test_step, total_rewards
