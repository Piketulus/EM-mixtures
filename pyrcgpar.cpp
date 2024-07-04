#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>

void logsumexp(torch::Tensor &gamma_Z, torch::Tensor &m) {
    m = torch::logsumexp(gamma_Z, 0);
    gamma_Z -= m;
}

torch::Tensor mixt_negnatgrad(torch::Tensor &gamma_Z, torch::Tensor &N_k, torch::Tensor &logl, torch::Tensor &dL_dphi) {
    torch::Tensor digamma_N_k = torch::digamma(N_k) - 1.0;
    dL_dphi.copy_((logl + digamma_N_k.unsqueeze(1)).sub_(gamma_Z));
    torch::Tensor temp = torch::exp(gamma_Z).mul_(dL_dphi);
    torch::Tensor colsums = torch::sum(temp, 0);
    torch::Tensor newnorm = torch::sum(temp.mul_(dL_dphi - colsums));
    return newnorm;
}

torch::Tensor update_N_k(torch::Tensor &gamma_Z, torch::Tensor &log_times_observed, torch::Tensor &alpha0) {
    torch::Tensor N_k = torch::sum((gamma_Z + log_times_observed).exp_(), 1);
    N_k += alpha0;
    return N_k;
}

torch::Tensor ELBO_rcg_mat(torch::Tensor &logl, torch::Tensor &gamma_Z, torch::Tensor &counts, torch::Tensor &N_k, torch::Tensor &bound_const) {
    torch::Tensor bound = torch::sum((gamma_Z + counts).exp_().mul_(logl - gamma_Z));
    bound += torch::sum(torch::lgamma(N_k));
    bound += bound_const;
    return bound;
}

torch::Tensor calc_bound_const(torch::Tensor &log_times_observed, torch::Tensor &alpha0) {
    torch::Tensor counts_sum = torch::sum(torch::exp(log_times_observed));
    torch::Tensor alpha0_sum = torch::sum(alpha0);
    torch::Tensor lgamma_alpha0_sum = torch::sum(torch::lgamma(alpha0));
    torch::Tensor bound_const = torch::lgamma(alpha0_sum) - torch::lgamma(alpha0_sum + counts_sum) - lgamma_alpha0_sum;
    return bound_const;
}

std::tuple<torch::Tensor, int> rcg_optl_mat(torch::Tensor &logl, torch::Tensor &log_times_observed, torch::Tensor &alpha0, double tol, int max_iters, torch::TensorOptions options) {
    int n_groups = logl.size(0);
    int n_obs = logl.size(1);

    torch::Tensor gamma_Z = torch::full({n_groups, n_obs}, std::log(1.0 / n_groups), options);
    torch::Tensor step = torch::zeros_like(logl, options);
    torch::Tensor oldstep = torch::zeros_like(logl, options);
    torch::Tensor oldm = torch::zeros({n_obs}, options);
    torch::Tensor oldnorm = torch::tensor(1.0, options);
    torch::Tensor bound = torch::tensor(-100000.0, options);
    torch::Tensor oldbound = torch::tensor(-100000.0, options);

    torch::Tensor bound_const = calc_bound_const(log_times_observed, alpha0);
    torch::Tensor N_k = update_N_k(gamma_Z, log_times_observed, alpha0);

    bool didreset = false;
    for (int k = 0; k < max_iters; ++k) {
        torch::Tensor newnorm = mixt_negnatgrad(gamma_Z, N_k, logl, step);
        torch::Tensor beta_FR = newnorm / oldnorm;
        oldnorm.copy_(newnorm);

        if (didreset) {
            oldstep.mul_(0.0);
        } else if (beta_FR.item<double>() > 0) {
            oldstep.mul_(beta_FR);
            step.add_(oldstep);
        }
        didreset = false;

        gamma_Z.add_(step);

        logsumexp(gamma_Z, oldm);
        N_k = update_N_k(gamma_Z, log_times_observed, alpha0);

        oldbound.copy_(bound);
        bound = ELBO_rcg_mat(logl, gamma_Z, log_times_observed, N_k, bound_const);

        if (bound.item<double>() < oldbound.item<double>()) {
            didreset = true;
            gamma_Z.add_(oldm); // revert step
            if (beta_FR.item<double>() > 0) {
                gamma_Z.sub_(oldstep);
            }

            logsumexp(gamma_Z, oldm);
            N_k = update_N_k(gamma_Z, log_times_observed, alpha0);

            bound = ELBO_rcg_mat(logl, gamma_Z, log_times_observed, N_k, bound_const);
        } else {
            oldstep.copy_(step);
        }

        if (k % 5 == 0) {
            std::cout << "iter: " << k << ", bound: " << bound.item<double>() << ", |g|: " << newnorm.item<double>() << std::endl;
        }

        if (bound.item<double>() - oldbound.item<double>() < tol && !didreset) {
            logsumexp(gamma_Z, oldm);
            return std::make_tuple(gamma_Z, k + 1);
        }

        if (newnorm.item<double>() < 0) {
            tol *= 10;
        }
    }

    logsumexp(gamma_Z, oldm);
    return std::make_tuple(gamma_Z, max_iters);
}

torch::Tensor mixture_components(torch::Tensor &probs, torch::Tensor &log_times_observed) {
    torch::Tensor n_times_total = torch::sum(torch::exp(log_times_observed));
    torch::Tensor thetas = torch::sum(torch::exp(probs + log_times_observed), 1) / n_times_total;
    return thetas;
}

void load_log_likelihoods_from_tsv(const std::string& file_path, std::vector<double>& log_likelihood_matrix, std::vector<double>& loglik_counts) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file");
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<double> row;
        std::string value;
        while (std::getline(iss, value, '\t')) {
            row.push_back(std::stof(value));
        }
        loglik_counts.push_back(std::log(row[0]));
        log_likelihood_matrix.insert(log_likelihood_matrix.end(), row.begin() + 1, row.end());
    }

    file.close();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];

    // Choose the device
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    if (device == torch::kCUDA) {
        std::cout << "Using GPU" << std::endl;
    }

    torch::Dtype precision = torch::kFloat64;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<double> log_likelihood_matrix;
    std::vector<double> loglik_counts;
    load_log_likelihoods_from_tsv(filename, log_likelihood_matrix, loglik_counts);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Loaded log likelihoods in " << std::chrono::duration<double>(end - start).count() << " seconds" << std::endl;

    // print size of log_likelihood_matrix in terms of memory
    std::cout << "Size of log_likelihood_matrix: " << log_likelihood_matrix.size() * sizeof(double) << " bytes" << std::endl;

    int num_rows = loglik_counts.size();
    int num_cols = log_likelihood_matrix.size() / num_rows;

    torch::TensorOptions options(precision);
    options = options.device(device);

    torch::Tensor logl = torch::from_blob((double*)log_likelihood_matrix.data(), {num_rows, num_cols}, precision).clone().to(device).t();
    torch::Tensor log_times_observed = torch::from_blob((double*)loglik_counts.data(), {num_rows}, precision).clone().to(device);
    torch::Tensor alpha0 = torch::ones({num_cols}, options);

    double tol = 1e-6;
    int max_iters = 5000;

    start = std::chrono::high_resolution_clock::now();

    auto [gamma_Z, num_iterations] = rcg_optl_mat(logl, log_times_observed, alpha0, tol, max_iters, options);
    torch::Tensor thetas = mixture_components(gamma_Z, log_times_observed);
    
    end = std::chrono::high_resolution_clock::now();

    std::cout << "Ran RCG algorithm in " << std::chrono::duration<double>(end - start).count() << " seconds" << std::endl;
    std::cout << "Converged in " << num_iterations << " iterations" << std::endl;

    std::cout << "Mixture components:" << std::endl;
    std::cout << thetas << std::endl;

    return 0;
}

