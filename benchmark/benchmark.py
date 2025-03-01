import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
import mne
import pandas as pd
from sklearn.decomposition import FastICA

def run_fastica(X, seed):
    """
    Runs FastICA on data X (shape: n_channels x n_samples).
    Returns estimated sources, unmixing matrix, and elapsed compute time.
    """
    ica = FastICA(random_state=seed, max_iter=1000)
    start = time.time()
    # FastICA expects data shape: (n_samples, n_features)
    S_est = ica.fit_transform(X.T)
    end = time.time()
    W_est = ica.components_
    return S_est, W_est, end - start

def run_infomax(X, seed):
    """
    Runs Infomax ICA using MNE's ICA on data X (shape: n_channels x n_samples).
    Converts X into an MNE RawArray before processing.
    Returns estimated sources, unmixing matrix, and elapsed compute time.
    """
    n_channels, n_samples = X.shape
    ch_names = [f"EEG{i+1}" for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types=ch_types)
    raw = mne.io.RawArray(X, info)
    
    ica = mne.preprocessing.ICA(n_components=n_channels, method='infomax', 
                                random_state=seed, max_iter=300)
    start = time.time()
    ica.fit(raw)
    end = time.time()
    W_est = ica.get_components()
    S_est = ica.get_sources(raw).get_data()
    return S_est, W_est, end - start

def run_sobi(X, seed, n_lags=10, tol=1e-6, max_iter=100):
    """
    Runs SOBI (Second-Order Blind Identification) on data X (shape: n_channels x n_samples).
    Returns estimated sources, unmixing matrix, and elapsed compute time.
    """
    np.random.seed(seed)
    start = time.time()
    n_channels, n_samples = X.shape

    # Whitening: compute covariance and eigen-decomposition
    C = np.cov(X)
    d, E = np.linalg.eigh(C)
    idx = np.argsort(d)[::-1]
    d = d[idx]
    E = E[:, idx]
    V = np.diag(1.0/np.sqrt(d)) @ E.T
    Z = V @ X

    # Compute lagged covariance matrices for lags 1 to n_lags
    R = []
    for tau in range(1, n_lags + 1):
        R_tau = (Z[:, tau:] @ Z[:, :-tau].T) / (n_samples - tau)
        R_tau = (R_tau + R_tau.T) / 2.0  # symmetrize
        R.append(R_tau)
    
    # Joint diagonalization via iterative Jacobi rotations
    U = np.eye(n_channels)
    for iteration in range(max_iter):
        off = 0.0
        for p in range(n_channels - 1):
            for q in range(p + 1, n_channels):
                g_sum = 0.0
                diff_sum = 0.0
                for R_tau in R:
                    g_sum += 2 * R_tau[p, q]
                    diff_sum += R_tau[p, p] - R_tau[q, q]
                theta = 0.5 * np.arctan2(g_sum, diff_sum + 1e-12)
                c = np.cos(theta)
                s = np.sin(theta)
                if np.abs(s) > tol:
                    temp_p = U[:, p].copy()
                    temp_q = U[:, q].copy()
                    U[:, p] = c * temp_p + s * temp_q
                    U[:, q] = -s * temp_p + c * temp_q
                    for k in range(len(R)):
                        Rp = R[k][p, :].copy()
                        Rq = R[k][q, :].copy()
                        R[k][p, :] = c * Rp + s * Rq
                        R[k][q, :] = -s * Rp + c * Rq
                        Rp = R[k][:, p].copy()
                        Rq = R[k][:, q].copy()
                        R[k][:, p] = c * Rp + s * Rq
                        R[k][:, q] = -s * Rp + c * Rq
                    off += np.abs(s)
        if off < tol:
            break

    W_est = U @ V
    S_est = W_est @ X
    end = time.time()
    return S_est, W_est, end - start

def experiment_varying_components(components, n_samples, n_runs):
    """
    Runs an experiment with varying number of components (with constant n_samples).
    Returns a dictionary with compute times for each algorithm.
    """
    algorithms = ['fastica', 'infomax', 'sobi']
    results = {alg: {n: [] for n in components} for alg in algorithms}
    for n_components in components:
        for run in range(n_runs):
            seed = run
            rng = np.random.RandomState(seed)
            # Generate independent source signals (Laplace distribution)
            S_true = rng.laplace(size=(n_components, n_samples))
            # Create random mixing matrix A and mix the sources: X = A * S_true
            A = rng.randn(n_components, n_components)
            X = np.dot(A, S_true)
            
            try:
                _, _, t_fastica = run_fastica(X, seed)
                results['fastica'][n_components].append(t_fastica)
            except Exception as e:
                print(f"FastICA failed for n_components={n_components}, run={run}: {e}")
            
            try:
                _, _, t_infomax = run_infomax(X, seed)
                results['infomax'][n_components].append(t_infomax)
            except Exception as e:
                print(f"Infomax failed for n_components={n_components}, run={run}: {e}")
            
            try:
                _, _, t_sobi = run_sobi(X, seed, n_lags=10)
                results['sobi'][n_components].append(t_sobi)
            except Exception as e:
                print(f"SOBI failed for n_components={n_components}, run={run}: {e}")
    return results

def experiment_varying_samples(constant_components, samples_list, n_runs):
    """
    Runs an experiment with constant number of components and varying number of samples.
    Returns a dictionary with compute times for each algorithm.
    """
    algorithms = ['fastica', 'infomax', 'sobi']
    results = {alg: {n: [] for n in samples_list} for alg in algorithms}
    for n_samples in samples_list:
        for run in range(n_runs):
            seed = run
            rng = np.random.RandomState(seed)
            S_true = rng.laplace(size=(constant_components, n_samples))
            A = rng.randn(constant_components, constant_components)
            X = np.dot(A, S_true)
            
            try:
                _, _, t_fastica = run_fastica(X, seed)
                results['fastica'][n_samples].append(t_fastica)
            except Exception as e:
                print(f"FastICA failed for n_samples={n_samples}, run={run}: {e}")
            
            try:
                _, _, t_infomax = run_infomax(X, seed)
                results['infomax'][n_samples].append(t_infomax)
            except Exception as e:
                print(f"Infomax failed for n_samples={n_samples}, run={run}: {e}")
            
            try:
                _, _, t_sobi = run_sobi(X, seed, n_lags=10)
                results['sobi'][n_samples].append(t_sobi)
            except Exception as e:
                print(f"SOBI failed for n_samples={n_samples}, run={run}: {e}")
    return results

def plot_experiment_results(x_values, results, xlabel, ylabel, title, filename, xscale="linear"):
    """
    Plots compute time (with error bars: mean ± std) vs. x_values for each algorithm.
    Saves the plot to the specified filename. The xscale can be set (e.g., 'log').
    """
    mean_times = {alg: [] for alg in results}
    std_times = {alg: [] for alg in results}
    for alg in results:
        for x in x_values:
            times = results[alg][x]
            mean_times[alg].append(np.mean(times))
            std_times[alg].append(np.std(times))
    plt.figure(figsize=(10, 6))
    for alg in results:
        plt.errorbar(x_values, mean_times[alg], yerr=std_times[alg],
                     capsize=5, marker='o', label=alg)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xscale(xscale)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved as '{filename}'.")

def save_results_to_csv(results, x_values, filename, x_label):
    """
    Saves the raw compute time results to a CSV file.
    The CSV will have columns: [x_label, algorithm, run, compute_time]
    """
    rows = []
    for alg in results:
        for x in x_values:
            times = results[alg][x]
            for run, t in enumerate(times):
                rows.append({x_label: x, "algorithm": alg, "run": run, "compute_time": t})
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Results saved as CSV to '{filename}'.")

def main():
    parser = argparse.ArgumentParser(
        description="ICA Performance Experiments: varying components and varying samples"
    )
    parser.add_argument('--n_runs', type=int, default=10,
                        help="Number of runs per configuration")
    parser.add_argument('--constant_samples', type=int, default=1000,
                        help="Constant number of samples for experiment 1 (varying components)")
    parser.add_argument('--constant_components', type=int, default=5,
                        help="Constant number of components for experiment 2 (varying samples)")
    args = parser.parse_args()
    
    # Experiment 1: Varying number of components (constant samples)
    components_list = list(range(3, 13))  # e.g., from 3 to 12 components
    results_components = experiment_varying_components(components_list, args.constant_samples, args.n_runs)
    plot_experiment_results(components_list, results_components,
                            xlabel="Number of Components",
                            ylabel="Compute Time (s)",
                            title="Compute Time vs. Number of Components",
                            filename="experiment_varying_components.png",
                            xscale="linear")
    save_results_to_csv(results_components, components_list, 
                        "results_varying_components.csv", "n_components")
    
    # Experiment 2: Varying number of samples (constant components)
    samples_list = [100, 500, 1000, 5000, 10000, 50000, 100000]  # e.g., varying samples
    results_samples = experiment_varying_samples(args.constant_components, samples_list, args.n_runs)
    plot_experiment_results(samples_list, results_samples,
                            xlabel="Number of Samples",
                            ylabel="Compute Time (s)",
                            title="Compute Time vs. Number of Samples",
                            filename="experiment_varying_samples.png",
                            xscale="log")
    save_results_to_csv(results_samples, samples_list,
                        "results_varying_samples.csv", "n_samples")

if __name__ == '__main__':
    main()
