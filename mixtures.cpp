#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <torch/torch.h>
#include <chrono>


// Function to load log likelihoods and counts from a TSV file
template <typename T>
void load_log_likelihoods_from_tsv(const std::string& file_path, std::vector<T>& log_likelihood_matrix, std::vector<T>& loglik_counts) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file");
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<T> row;
        std::string value;
        while (std::getline(iss, value, '\t')) {
            row.push_back(std::stof(value));
        }
        loglik_counts.push_back(std::log(row[0]));
        log_likelihood_matrix.insert(log_likelihood_matrix.end(), row.begin() + 1, row.end());
    }

    file.close();
}

// EM algorithm function
template <typename T>
std::pair<torch::Tensor, int> em_algorithm(const std::vector<T>& log_likelihood_matrix, const std::vector<T>& loglik_counts, torch::ScalarType dtype, T threshold = 1e-6, int max_iters = 5000, bool use_gpu = true) {

    torch::Device device = (torch::cuda::is_available() && use_gpu) ? torch::kCUDA : torch::kCPU;
    if (device == torch::kCUDA) {
        std::cout << "Using CUDA" << std::endl;
    }

    int num_rows = loglik_counts.size();
    int num_cols = log_likelihood_matrix.size() / num_rows;

    torch::Tensor log_likelihoods = torch::from_blob((T*)log_likelihood_matrix.data(), {num_rows, num_cols}, dtype).clone().to(device);
    torch::Tensor loglik_counts_tensor = torch::from_blob((T*)loglik_counts.data(), {num_rows}, dtype).clone().to(device);

    // pre allocate log_weighted_likelihoods
    torch::Tensor log_weighted_likelihoods = torch::empty({num_rows, num_cols}, dtype).to(device);

    torch::Tensor theta = torch::ones({num_cols}, dtype).to(device) / num_cols;
    torch::Tensor prev_loss = torch::tensor(std::numeric_limits<T>::infinity(), dtype).to(device);
    torch::Tensor threshold_tensor = torch::tensor(threshold, dtype).to(device);

    int iteration;
    for (iteration = 0; iteration < max_iters; ++iteration) {
        // E-step: Compute responsibilities
        log_weighted_likelihoods.copy_(log_likelihoods + torch::log(theta));
        torch::Tensor log_sum_exp = torch::logsumexp(log_weighted_likelihoods, 1, true);
        //torch::Tensor log_responsibilities = log_weighted_likelihoods.sub_(log_sum_exp);
        log_weighted_likelihoods.sub_(log_sum_exp);

        // M-step: Update theta values weighted by log counts
        //torch::Tensor log_weighted_responsibilities = log_responsibilities.add_(loglik_counts_tensor.unsqueeze(1));
        log_weighted_likelihoods.add_(loglik_counts_tensor.unsqueeze(1));
        //log_weighted_responsibilities.exp_();
        log_weighted_likelihoods.exp_();
        torch::Tensor new_theta = log_weighted_likelihoods.sum(0) / torch::exp(loglik_counts_tensor).sum();

        // Compute the log likelihood
        torch::Tensor log_likelihood = torch::sum((log_sum_exp.add_(loglik_counts_tensor.unsqueeze(1))).exp_());

        // Check for convergence
        torch::Tensor loss = -log_likelihood;
        if (torch::abs(prev_loss - loss).lt(threshold_tensor).item<bool>()) {
            break;
        }
        prev_loss = loss;

        // Update theta
        theta.copy_(new_theta);
    }

    return {theta.cpu(), iteration + 1};
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <filename> <float|double>" << std::endl;
        return 1;
    }
    std::string filename = argv[1];
    std::string precision = (argc > 2) ? argv[2] : "double";

    torch::ScalarType dtype;
    if (precision == "double") {
        dtype = torch::kFloat64;
    } else {
        dtype = torch::kFloat32;
    }

    if (precision == "double") {
        using T = double;

        std::vector<T> log_likelihood_matrix;
        std::vector<T> loglik_counts;

        auto start = std::chrono::high_resolution_clock::now();
        
        load_log_likelihoods_from_tsv(filename, log_likelihood_matrix, loglik_counts);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> load_time = end - start;
        std::cout << "Loaded log likelihoods in " << load_time.count() << " seconds" << std::endl;

        torch::Tensor theta;
        int num_iterations;

        start = std::chrono::high_resolution_clock::now();

        auto result = em_algorithm(log_likelihood_matrix, loglik_counts, dtype, 1e-3);
        theta = result.first;
        num_iterations = result.second;

        end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> em_time = end - start;

        std::cout << "Ran EM algorithm in " << em_time.count() << " seconds" << std::endl;
        std::cout << "Number of iterations: " << num_iterations << std::endl;

        std::cout << "Theta: " << std::endl;
        std::cout << theta << std::endl;
    } else {
        using T = float;

        std::vector<T> log_likelihood_matrix;
        std::vector<T> loglik_counts;

        auto start = std::chrono::high_resolution_clock::now();

        load_log_likelihoods_from_tsv(filename, log_likelihood_matrix, loglik_counts);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> load_time = end - start;
        std::cout << "Loaded log likelihoods in " << load_time.count() << " seconds" << std::endl;

        torch::Tensor theta;
        int num_iterations;

        start = std::chrono::high_resolution_clock::now();

        auto result = em_algorithm(log_likelihood_matrix, loglik_counts, dtype);
        theta = result.first;
        num_iterations = result.second;

        end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> em_time = end - start;

        std::cout << "Ran EM algorithm in " << em_time.count() << " seconds" << std::endl;
        std::cout << "Number of iterations: " << num_iterations << std::endl;

        std::cout << "Theta: " << std::endl;
        std::cout << theta << std::endl;
    }

    return 0;
}
