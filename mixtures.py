import numpy as np
import torch
import sys
import time

def load_log_likelihoods_from_tsv(file_path):
    """
    Load log likelihoods and counts from a TSV file.

    Args:
        file_path (str): The path to the TSV file.

    Returns:
        log_likelihood_matrix (ndarray): The log likelihoods matrix (unique observations x groups).
        loglik_counts (ndarray): The counts of each unique observation.
    """
    data = np.loadtxt(file_path, delimiter='\t')
    loglik_counts = data[:, 0]
    log_likelihood_matrix = data[:, 1:]
    loglik_counts = np.log(loglik_counts)

    return log_likelihood_matrix, loglik_counts


def em_algorithm(log_likelihood_matrix, loglik_counts, threshold=1e-6, max_iters=5000, use_gpu=True):
    """
    Perform the EM algorithm to infer the mixing proportions given log likelihoods and counts.

    Args:
        log_likelihood_matrix (ndarray or tensor): The log likelihoods matrix (unique observations x groups).
        loglik_counts (ndarray or tensor): The counts of each unique observation.
        threshold (float): The convergence threshold.
        max_iters (int): The maximum number of iterations.
        use_gpu (bool): Whether to use GPU.

    Returns:
        theta (ndarray): The inferred mixing proportions for each group.
    """

    data_type = torch.float32 # data type for the tensors

    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    #print if using cuda
    if device == torch.device("cuda"):
        print("Using CUDA")

    log_likelihoods = torch.tensor(log_likelihood_matrix, dtype=data_type, device=device)
    loglik_counts = torch.tensor(loglik_counts, dtype=data_type, device=device)

    num_unique_obs, num_groups = log_likelihoods.shape

    # Pre-allocate tensors
    log_weighted_likelihoods = torch.empty(num_unique_obs, num_groups, dtype=data_type, device=device)
    log_sum_exp = torch.empty(num_unique_obs, 1, dtype=data_type, device=device)
    log_responsibilities = torch.empty(num_unique_obs, num_groups, dtype=data_type, device=device)
    log_weighted_responsibilities = torch.empty(num_unique_obs, num_groups, dtype=data_type, device=device)

    # Initialize mixing proportions (theta values)
    theta = torch.ones(num_groups, dtype=data_type, device=device) / num_groups  # uniform initialization

    prev_loss = torch.tensor(float('inf'), dtype=data_type, device=device)

    threshold_tensor = torch.tensor(threshold, dtype=data_type, device=device)

    for iteration in range(max_iters):
        # E-step: Compute responsibilities
        log_weighted_likelihoods = log_likelihoods + torch.log(theta)
        log_sum_exp = torch.logsumexp(log_weighted_likelihoods, dim=1, keepdim=True)
        log_responsibilities = log_weighted_likelihoods - log_sum_exp

        # M-step: Update theta values weighted by log counts
        log_weighted_responsibilities = log_responsibilities + loglik_counts.unsqueeze(1)
        torch.exp_(log_weighted_responsibilities)
        new_theta = log_weighted_responsibilities.sum(dim=0) / torch.sum(torch.exp(loglik_counts))

        # Compute the log likelihood
        log_likelihood = torch.sum(torch.exp_(log_sum_exp + loglik_counts.unsqueeze(1)))

        # Check for convergence
        loss = -log_likelihood
        if abs(prev_loss - loss) < threshold_tensor:
            break
        prev_loss = loss

        # Update theta
        theta.copy_(new_theta)

    return theta.detach().cpu().numpy(), iteration + 1


if __name__ == '__main__':
    # Get filename from command line arguments
    if len(sys.argv) < 2:
        print("Usage: python mixtures.py <filename>")
        sys.exit(1)
    filename = sys.argv[1]

    # Time loading the log likelihoods
    start = time.time()

    # Load the log likelihoods and counts from the TSV file
    log_likelihood_matrix, loglik_counts = load_log_likelihoods_from_tsv(filename)

    end = time.time()

    print(f"Loaded log likelihoods in {end - start} seconds")

    # Time the EM algorithm
    start = time.time()

    # Run the EM algorithm to infer mixing proportions
    theta, num_iterations = em_algorithm(log_likelihood_matrix, loglik_counts)

    end = time.time()

    print(f"Ran EM algorithm in {end - start} seconds")

    print(f"Converged in {num_iterations} iterations")
    print("Mixing proportions (theta values):", theta)
