import numpy as np

def stable_sigmoid(x):
    # Clip x to avoid overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def infoMaxICA(X, lr=0.001, max_iter=1000, tol=1e-5):
    """
    Perform ICA using the InfoMax algorithm with stability improvements.
    
    Parameters:
      X       : np.ndarray of shape (n_components, n_samples) (assumed to be centered and whitened)
      lr      : Learning rate (step size)
      max_iter: Maximum number of iterations
      tol     : Tolerance for convergence
      
    Returns:
      W       : The estimated unmixing matrix.
      S       : The separated (independent) components, S = W @ X.
    """
    n, T = X.shape
    # Initialize W as a random matrix
    W = np.random.randn(n, n)
    
    for i in range(max_iter):
        # Compute estimated sources
        Y = W @ X  # Shape: (n, T)
        # Use the stable sigmoid
        gY = stable_sigmoid(Y)
        # InfoMax update rule
        dW = lr * (np.eye(n) + (1 - 2 * gY) @ Y.T) @ W
        W_new = W + dW
        
        # Normalize rows of W_new to help with numerical stability
        W_new = W_new / np.linalg.norm(W_new, axis=1, keepdims=True)
        
        # Check for convergence
        if np.linalg.norm(dW) < tol:
            print(f"Converged at iteration {i}")
            W = W_new
            break
        
        W = W_new
    
    S = W @ X
    return W, S
