import numpy as np
from collections import defaultdict

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for state in range(env.nS): 
            old_v = V[state]
            temp_v = 0
            for action in range(env.nA): 
                # Get transition probability for each action from state
                [trans_prob] = env.P[state][action]
                prob = trans_prob[0] # Use policy given 
                reward = trans_prob[2]
                next_state = trans_prob[1]
                # Question: what to do with 'done'
                temp_v += policy[state, action] * prob * (reward + discount_factor*V[next_state]) # Sum for all actions
            V[state] = temp_v
            delta = max(delta, np.abs(old_v - V[state]))
        if delta < theta:
            return np.array(V)

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    count = 0
    while True:
        # Policy evaluation 
        V = policy_eval_v(policy, env, discount_factor)
        # Policy improvement 
        policy_stable = True
        for state in range(env.nS):
            old_action_array = policy[state].copy()
            old_action = np.argmax(old_action_array)
            action_array = np.zeros(env.nA)
            for action in range(env.nA): 
                [trans_prob] = env.P[state][action]
                prob = trans_prob[0] 
                reward = trans_prob[2]
                next_state = trans_prob[1]
                action_array[action] += prob * (reward + discount_factor*V[next_state]) 
            new_action = np.argmax(action_array)
            if old_action != new_action: # policy iteration is deterministic in book.. hmm.. 
                policy_stable = False
            policy[state] = np.zeros(env.nA)
            policy[state][new_action] = 1. 
        if policy_stable:
            return policy, V
        print("iteration:", count)
        count += 1 

def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.        
    """
    
    # Start with an all 0 Q-value function
    Q = np.zeros((env.nS, env.nA))
    # YOUR CODE HERE
    while True:
        delta = 0
        for s in range(env.nS): 
            old_q = Q[s]
            temp_q = np.zeros(env.nA)
            for a in range(env.nA):
                r = env.P[s][a][0][2]
                s1 = env.P[s][a][0][1]
                temp_q[a] = (r + discount_factor * np.max(Q[s1]))
                delta = max(delta, np.abs(old_q[a] - temp_q[a]))
            Q[s] = temp_q
        if delta < theta:
            break
    policy = np.zeros((env.nS, env.nA))
    policy[np.arange(env.nS), np.argmax(Q, axis=1)] = 1
    return policy, Q
