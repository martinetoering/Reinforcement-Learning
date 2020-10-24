import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class NNPolicy(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        """
        Performs a forward pass through the network.
        
        Args:
            x: input tensor (first dimension is a batch dimension)
            
        Return:
            Probabilities of performing all actions in given input states x. Shape: batch_size x action_space_size
        """
        # YOUR CODE HERE
        hidden_values = F.relu(self.l1(x))
        out = F.softmax(self.l2(hidden_values), dim=1)
        return out
        
    def get_probs(self, obs, actions):
        """
        This function takes a tensor of states and a tensor of actions and returns a tensor that contains 
        a probability of perfoming corresponding action in all states (one for every state action pair). 

        Args:
            obs: a tensor of states. Shape: batch_size x obs_dim
            actions: a tensor of actions. Shape: batch_size x 1

        Returns:
            A torch tensor filled with probabilities. Shape: batch_size x 1.
        """
        # YOUR CODE HERE
        all_probs = self.forward(obs)
        action_probs = all_probs.gather(1, actions)
        return action_probs
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: state as a tensor. Shape: 1 x obs_dim or obs_dim

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        if obs.dim() < 2:
            obs = obs.unsqueeze(0)
        probs = self.forward(obs)
        action = torch.multinomial(probs.squeeze(), 1).item()
        return action
        
        

def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function as tensors.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of tensors (states, actions, rewards, dones). All tensors should have same first dimension and 
        should have dim=2. This means that vectors of length N (states, rewards, actions) should be Nx1.
        Hint: Do not include the state after termination in states.
    """
    states = []
    actions = []
    rewards = []
    dones = []
    
    # YOUR CODE HERE
    state = torch.from_numpy(env.reset()).to(dtype=torch.float)
    states.append(state)
    
    while True:
        with torch.no_grad():
            action = policy.sample_action(state)
        observation, reward, done, info = env.step(action)

        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        
        if done:
            break
        
        obs = torch.from_numpy(observation).to(dtype=torch.float)
        states.append(obs)
        state = obs
        
    states = torch.stack(states, dim=0)
    actions = torch.tensor(actions).unsqueeze(1)
    rewards = torch.tensor(rewards).unsqueeze(1)
    dones = torch.tensor(dones).unsqueeze(1)
    return states, actions, rewards, dones

def compute_reinforce_loss(policy, episode, discount_factor):
    """
    Computes reinforce loss for given episode.

    Args:
        policy: A policy which allows us to get probabilities of actions in states with its get_probs method.

    Returns:
        loss: reinforce loss
    """
    # Compute the reinforce loss
    # Make sure that your function runs in LINEAR TIME
    # Note that the rewards/returns should be maximized 
    # while the loss should be minimized so you need a - somewhere
    # YOUR CODE HERE
    
    states, actions, rewards, dones = episode
    ep_len = len(rewards)
    probs = policy.get_probs(states, actions)
    log_probs = torch.log(probs)
    discounts = torch.pow(torch.tensor(discount_factor).repeat(ep_len, 1), torch.arange(ep_len).unsqueeze(1))
    loss = -1 * (log_probs * ((discounts * rewards).flip(0)).cumsum(0).flip(0)).sum()
    return loss

# YOUR CODE HERE
# raise NotImplementedError

def run_episodes_policy_gradient(policy, env, num_episodes, discount_factor, learn_rate, 
                                 sampling_function=sample_episode):
    optimizer = optim.Adam(policy.parameters(), learn_rate)
    
    episode_durations = []
    for i in range(num_episodes):
        
        # YOUR CODE HERE
        optimizer.zero_grad()
    
        episode = sampling_function(env, policy)
            
        loss = compute_reinforce_loss(policy, episode, discount_factor)
        loss.backward()
        optimizer.step()
                           
        if i % 10 == 0:
            print("{2} Episode {0} finished after {1} steps"
                  .format(i, len(episode[0]), '\033[92m' if len(episode[0]) >= 195 else '\033[99m'))
        episode_durations.append(len(episode[0]))
        
    return episode_durations
