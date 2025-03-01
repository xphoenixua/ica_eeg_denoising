import numpy as np


def whiten(X):
    """
    Whiten the observed signals.
    Input:
      X: array of shape (n_channels, n_samples)
    Returns:
      X_white: whitened data
      whitening_matrix: matrix used for whitening
    """

    X_centered = X - np.mean(X, axis=1, keepdims=True)

    cov = np.cov(X_centered)

    d, E = np.linalg.eigh(cov)

    idx = d.argsort()[::-1]
    d = d[idx]
    E = E[:, idx]

    D_inv = np.diag(1.0 / np.sqrt(d))
    whitening_matrix = D_inv @ E.T
    X_white = whitening_matrix @ X_centered
    return X_white, whitening_matrix


def compute_covariance_matrices(X, max_lag):
    """
    Compute covariance matrices at multiple time delays for the whitened signals.
    Input:
      X: whitened data of shape (n_channels, n_samples)
      max_lag: maximum time delay (tau) to use
    Returns:
      cov_matrices: list of covariance matrices for lags 1 ... max_lag
    """
    n_channels, n_samples = X.shape
    cov_matrices = []
    for tau in range(1, max_lag + 1):
        X1 = X[:, tau:]
        X2 = X[:, :n_samples - tau]
        R_tau = (X1 @ X2.T) / (n_samples - tau)
        R_tau = (R_tau + R_tau.T) / 2.0
        cov_matrices.append(R_tau)
    return cov_matrices


def joint_diagonalization(cov_matrices, eps=1e-6, max_iter=1000):
    """
    Joint diagonalization via iterative Givens rotations.
    Input:
      cov_matrices: list of covariance matrices to be jointly diagonalized
      eps: convergence threshold
      max_iter: maximum number of iterations
    Returns:
      V: the joint diagonalizer matrix
    """
    n_channels = cov_matrices[0].shape[0]
    V = np.eye(n_channels)

    for iteration in range(max_iter):
        change = 0
        for p in range(n_channels - 1):
            for q in range(p + 1, n_channels):
                g_sum = 0.0
                h_sum = 0.0
                for R in cov_matrices:
                    g_pp = R[p, p]
                    g_qq = R[q, q]
                    g_pq = R[p, q]

                    g_sum += 2 * g_pq
                    h_sum += (g_qq - g_pp)

                phi = 0.5 * np.arctan2(g_sum, h_sum + 1e-12)
                c = np.cos(phi)
                s = np.sin(phi)
                if np.abs(s) < eps:
                    continue

                J = np.eye(n_channels)
                J[p, p] = c
                J[q, q] = c
                J[p, q] = -s
                J[q, p] = s

                V = V @ J

                for i in range(len(cov_matrices)):
                    R = cov_matrices[i]
                    Rp = R[p, :].copy()
                    Rq = R[q, :].copy()
                    R[p, :] = c * Rp - s * Rq
                    R[q, :] = s * Rp + c * Rq

                    Rp = R[:, p].copy()
                    Rq = R[:, q].copy()
                    R[:, p] = c * Rp - s * Rq
                    R[:, q] = s * Rp + c * Rq
                    cov_matrices[i] = R
                change += np.abs(s)
        if change < eps:
            break
    return V


def sobi(X, max_lag=100):
    """
    Apply the SOBI algorithm to separate mixed signals.
    Input:
      X: observed data (n_channels x n_samples)
      max_lag: maximum time delay to consider for covariance estimation
    Returns:
      W: the matrix used to recover the sources
      S: estimated source signals
    """
    X_white, whitening_matrix = whiten(X)
    cov_matrices = compute_covariance_matrices(X_white, max_lag)
    U = joint_diagonalization(cov_matrices)
    W = U.T @ whitening_matrix
    S = W @ X
    return W, S