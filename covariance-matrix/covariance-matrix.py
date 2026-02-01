import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.asarray(X)
    N = X.shape[0]
    
    if X.size == 0 or X.ndim < 2 or X.shape[0] <= 1:
        return None

    mu = np.mean(X, axis = 0)
    X_centered =  X - mu

    covariance = (X_centered.T @ X_centered) / (N - 1)
    return covariance
