import numpy as np
from collections import defaultdict
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class SimpleBlackjackPolicy(object):
    """
    A simple BlackJack policy that sticks with 20 or 21 points and hits otherwise.
    """
    def get_probs(self, states, actions):
        """
        This method takes a list of states and a list of actions and returns a numpy array that contains a probability
        of perfoming action in given state for every corresponding state action pair. 

        Args:
            states: a list of states.
            actions: a list of actions.

        Returns:
            Numpy array filled with probabilities (same length as states and actions)
        """
        probs = np.zeros(len(states))
        for i in range(len(states)):
            if states[i][0] == 20 or states[i][0] == 21: # If sum is 20 or 21
                if actions[i] == 0:
                    probs[i] = 1. # Action 0 (sick) 100% prob, so 0% action 1
                else:
                    probs[i] = 0.
            else:
                if actions[i] == 0:
                    probs[i] = 0.
                else:
                    probs[i] = 1. # Action 1 (hit) 100% prob
                
        return np.array(probs)
    
    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
        actions = [0, 1]
        probs = self.get_probs([state, state], actions)
        [action] = np.random.choice(actions, 1, p=probs)
        return action

def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function and policy's sample_action function as lists.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of lists (states, actions, rewards, dones). All lists should have same length. 
        Hint: Do not include the state after the termination in the list of states.
    """
    
    states = []
    actions = []
    rewards = []
    dones = []
    
    state = env.reset()
    states.append(state)
    
    while True:
        
        action = policy.sample_action(state)
        observation, reward, done, info = env.step(action)

        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        
        if done:
            return states, actions, rewards, dones
        
        states.append(observation)
        state = observation

def mc_prediction(env, policy, num_episodes, discount_factor=1.0, sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    
    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    returns_count = defaultdict(float)
    
    returns_total = defaultdict(float)
    
    for i in tqdm(range(num_episodes)):
        
        states, actions, rewards, dones = sampling_function(env, policy)
        G = 0 
        
        for step in range(len(actions)-1, -1, -1): # A_{t-1}, R_{t}
            
            state = states[step]
            G = discount_factor*G + rewards[step]
            
            if state not in states[0:step]:
                returns_total[state] += G
                returns_count[state] += 1
                V[state] = returns_total[state]/returns_count[state]
        
    return V

class RandomBlackjackPolicy(object):
    """
    A random BlackJack policy.
    """
    def get_probs(self, states, actions):
        """
        This method takes a list of states and a list of actions and returns a numpy array that contains 
        a probability of perfoming action in given state for every corresponding state action pair. 

        Args:
            states: a list of states.
            actions: a list of actions.

        Returns:
            Numpy array filled with probabilities (same length as states and actions)
        """
        probs = np.zeros(len(states))
        probs.fill(0.5)
        return np.array(probs)
    
    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
        actions = [0, 1]
        probs = self.get_probs([state, state], actions)
        [action] = np.random.choice(actions, 1, p=probs)
        return action

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    returns_count = defaultdict(float)
    
    returns_total = defaultdict(float)
    
    for i in tqdm(range(num_episodes)):

        states, actions, rewards, dones = sampling_function(env, behavior_policy)
        G = 0.
        W = 1.
                
        for step in range(len(actions)-1, -1, -1): # A_{t-1}, R_{t}
            
            state = states[step]
            G = discount_factor*G + rewards[step]

            returns_count[state] += 1       
            

            V[state] = V[state] + (((W * G) - V[state]) / returns_count[state])
            
            
                                    
            if W == 0:
                break
                
            W = W * (target_policy.get_probs([state], [actions[step]]) / behavior_policy.get_probs([state], [actions[step]]))        
    return V
