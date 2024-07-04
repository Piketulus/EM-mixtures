import numpy as np
import torch
import sys
import time


def logsumexp(gamma_Z):
    m = torch.logsumexp(gamma_Z, dim=0)
    gamma_Z -= m
    return m

def mixt_negnatgrad(gamma_Z, N_k, logl, dL_dphi):
    digamma_N_k = torch.digamma(N_k) - 1.0
    dL_dphi.copy_((logl + digamma_N_k.unsqueeze(1)).sub_(gamma_Z))
    temp = torch.exp(gamma_Z).mul_(dL_dphi)
    colsums = torch.sum(temp, dim=0)
    newnorm = torch.sum(temp.mul_(dL_dphi - colsums))
    return newnorm

def update_N_k(gamma_Z, log_times_observed, alpha0):
    #N_k = torch.exp(gamma_Z).matmul(torch.exp(log_times_observed))
    N_k = torch.sum(torch.exp_(gamma_Z + log_times_observed), dim=1)
    N_k += alpha0
    return N_k

def ELBO_rcg_mat(logl, gamma_Z, counts, N_k, bound_const):
    bound = torch.sum(torch.exp_(gamma_Z + counts).mul_(logl - gamma_Z))
    bound += torch.sum(torch.lgamma(N_k))
    bound += bound_const
    return bound

def calc_bound_const(log_times_observed, alpha0):
    counts_sum = torch.sum(torch.exp(log_times_observed))
    alpha0_sum = torch.sum(alpha0)
    lgamma_alpha0_sum = torch.sum(torch.lgamma(alpha0))
    bound_const = torch.lgamma(alpha0_sum) - torch.lgamma(alpha0_sum + counts_sum) - lgamma_alpha0_sum
    return bound_const

def rcg_optl_mat(logl, log_times_observed, alpha0, tol, max_iters, precision=torch.float64):
    n_groups, n_obs = logl.shape

    gamma_Z = torch.full((n_groups, n_obs), np.log(1.0 / n_groups), dtype=precision, device=logl.device)
    step = torch.zeros_like(logl, dtype=precision, device=logl.device)
    oldstep = torch.zeros_like(logl, dtype=precision, device=logl.device)
    oldm = torch.zeros(n_obs, dtype=precision, device=logl.device)
    oldnorm = torch.tensor(1.0, dtype=precision, device=logl.device)
    bound = torch.tensor(-100000.0, dtype=precision, device=logl.device)
    oldbound = torch.tensor(-100000.0, dtype=precision, device=logl.device)

    bound_const = calc_bound_const(log_times_observed, alpha0)

    N_k = update_N_k(gamma_Z, log_times_observed, alpha0)

    didreset = False
    for k in range(max_iters):
        newnorm = mixt_negnatgrad(gamma_Z, N_k, logl, step)
        beta_FR = newnorm / oldnorm
        oldnorm.copy_(newnorm)
        
        if didreset:
            oldstep.mul_(0.0)
        elif beta_FR > 0:
            oldstep.mul_(beta_FR)
            step.add_(oldstep)
        didreset = False

        gamma_Z.add_(step)

        oldm = logsumexp(gamma_Z)
        N_k = update_N_k(gamma_Z, log_times_observed, alpha0)

        oldbound.copy_(bound)
        bound = ELBO_rcg_mat(logl, gamma_Z, log_times_observed, N_k, bound_const)

        if bound < oldbound:
            didreset = True
            gamma_Z.add_(oldm) # revert step
            if beta_FR > 0:
                gamma_Z.sub_(oldstep)

            oldm = logsumexp(gamma_Z)
            N_k = update_N_k(gamma_Z, log_times_observed, alpha0)

            bound = ELBO_rcg_mat(logl, gamma_Z, log_times_observed, N_k, bound_const)
        else:
            oldstep.copy_(step)
        
        if k % 5 == 0:
            print(f"iter: {k}, bound: {bound.item()}, |g|: {newnorm.item()}")

        if bound - oldbound < tol and not didreset:
            oldm = logsumexp(gamma_Z)
            return gamma_Z, k + 1

        if newnorm < 0:
            tol *= 10

    oldm = logsumexp(gamma_Z)
    return gamma_Z, max_iters

def load_log_likelihoods_from_tsv(file_path):
    data = np.loadtxt(file_path, delimiter='\t')
    loglik_counts = data[:, 0]
    log_likelihood_matrix = data[:, 1:]
    return log_likelihood_matrix, loglik_counts

def mixture_components(probs, log_times_observed):
    n_times_total = torch.sum(torch.exp(log_times_observed))
    thetas = torch.sum(torch.exp_(probs + log_times_observed), dim=1) / n_times_total
    return thetas

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python pyrcgpar.py <filename>")
        sys.exit(1)
    filename = sys.argv[1]

    # choose the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print("Using GPU")

    precision=torch.float64

    start = time.time()
    log_likelihood_matrix, loglik_counts = load_log_likelihoods_from_tsv(filename)
    end = time.time()
    print(f"Loaded log likelihoods in {end - start} seconds")

    log_likelihoods = torch.tensor(log_likelihood_matrix, dtype=precision, device=device).T # transpose
    loglik_counts = torch.log_(torch.tensor(loglik_counts, dtype=precision, device=device))
    alpha0 = torch.ones(log_likelihoods.shape[0], dtype=precision, device=device)

    print(f"Size of log_likelihoods: {log_likelihoods.numel() * log_likelihoods.element_size()} bytes")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated()} bytes")

    tol = 1e-6
    max_iters = 5000

    start = time.time()
    gamma_Z, num_iterations = rcg_optl_mat(log_likelihoods, loglik_counts, alpha0, tol, max_iters, precision)
    thetas = mixture_components(gamma_Z, loglik_counts)
    end = time.time()

    print(f"Max memory allocated: {torch.cuda.max_memory_allocated()} bytes")

    print(f"Ran RCG algorithm in {end - start} seconds")
    print(f"Converged in {num_iterations} iterations")

    print("Mixture components:")
    print(thetas)
