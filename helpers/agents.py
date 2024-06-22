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
        env (gymnasium.Env): The environment to interact with.'''

    def __init__(self, env):
        self.env = env

    def get_action(self):
        '''Get a random action from the environment's action space.
        
        Returns:
            action (int): A random action.
            
        '''
        return self.env.action_space.sample()