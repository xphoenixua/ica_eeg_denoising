import numpy as np


def fastICA(X, n_components, tol=1e-5, max_iter=1000):
    """
    FastICA implementation using fixed-point iteration (manual implementation).

    Parameters:
        X : array-like, shape (n_samples, n_features)
            The observed mixed signals.
        n_components : int
            The number of independent components to extract.
        tol : float, optional
            Convergence tolerance.
        max_iter : int, optional
            Maximum number of iterations per component.

    Returns:
        S : array-like, shape (n_samples, n_components)
            Estimated source signals.
        W : array-like, shape (n_components, n_features)
            Estimated unmixing matrix.
    """
    X_centered = X - np.mean(X, axis=0)

    cov = np.cov(X_centered, rowvar=False)

    d, E = np.linalg.eigh(cov)

    idx = np.argsort(d)[::-1]
    d = d[idx]
    E = E[:, idx]

    D_inv = np.diag(1.0 / np.sqrt(d))
    whitening_matrix = D_inv @ E.T
    X_whitened = X_centered @ whitening_matrix.T

    n_samples, n_features = X_whitened.shape
    W = np.zeros((n_components, n_features))

    for i in range(n_components):
        w = np.random.rand(n_features)
        w /= np.linalg.norm(w)

        for iteration in range(max_iter):
            wx = np.dot(X_whitened, w)
            g = np.tanh(wx)
            g_prime = 1 - np.tanh(wx) ** 2

            w_new = np.mean(X_whitened * g[:, np.newaxis], axis=0) - np.mean(g_prime) * w

            if i > 0:
                w_new -= np.dot(np.dot(W[:i], w_new), W[:i])

            w_new /= np.linalg.norm(w_new)

            if np.abs(np.abs(np.dot(w_new, w)) - 1) < tol:
                break
            w = w_new

        W[i, :] = w_new

    S = np.dot(X_whitened, W.T)
    return W, S
