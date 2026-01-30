import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """
    # Write code here
    n = len(rewards)
    returns = np.zeros(n)
    advantages = np.zeros(n)

    running = 0
    for t in reversed(range(n)):
        running = rewards[t] + gamma * running
        returns[t] = running
        advantages[t] = returns[t] - V[t]
    return advantages
