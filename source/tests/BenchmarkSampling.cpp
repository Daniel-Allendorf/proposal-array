#include <cstdint>
#include <numbers>
#include <random>
#include <sampling/ScopedTimer.hpp>
#include <sampling/AliasTable.hpp>
#include <sampling/ProposalArray.hpp>
#include <sampling/BinaryTree.hpp>

using namespace sampling;

std::vector<double> generate_noisy_uniform_weights(size_t n, std::mt19937_64& gen) {
    std::uniform_real_distribution<double> weight_dist(0, n);
    std::vector<double> weights;
    weights.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        double random_weight = weight_dist(gen);
        weights.push_back(random_weight);
    }
    return weights;
}

std::vector<double> generate_power_law_weights(size_t n, std::mt19937_64& gen) {
    std::uniform_real_distribution<double> real_dist(0, 1);
    std::vector<double> weights;
    weights.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        double C = std::numbers::pi * std::numbers::pi / 6;
        size_t w = 1;
        double ww = 1.;
        while (real_dist(gen) > ww / C) {
            C -= ww;
            w++;
            ww = 1. / (w * w);
        }
        weights.push_back(w);
    }
    return weights;
}

std::vector<double> generate_noisy_delta_weights(size_t n, std::mt19937_64& gen) {
    std::uniform_real_distribution<double> weight_dist(0, 1);
    std::vector<double> weights;
    weights.reserve(n);
    for (size_t i = 0; i < n - 1; ++i) {
        double random_weight = weight_dist(gen);
        weights.push_back(random_weight);
    }
    weights.push_back(n);
    return weights;
}

void benchmark_at_sampling(const std::vector<double>& weights, size_t samples, std::mt19937_64& gen, std::string name) {
    AliasTable at(weights);
    {
        tools::ScopedTimer timer("AliasTable " + name + " [n: " + std::to_string(weights.size()) + "]");
        for (size_t s = 0; s < samples; ++s) {
            volatile size_t sample = at.sample(gen);
        }
    }
}

void benchmark_pa_sampling(const std::vector<double>& weights, size_t samples, std::mt19937_64& gen, std::string name) {
    ProposalArray pa(weights);
    {
        tools::ScopedTimer timer("ProposalArray " + name + " [n: " + std::to_string(weights.size()) + "]");
        for (size_t s = 0; s < samples; ++s) {
            volatile size_t sample = pa.sample(gen);
        }
    }
}

void benchmark_dd_sampling(const std::vector<double>& weights, size_t samples, std::mt19937_64& gen, std::string name) {
    std::discrete_distribution<size_t> dd(weights.begin(), weights.end());
    {
        tools::ScopedTimer timer("DiscreteDistribution " + name + " [n: " + std::to_string(weights.size()) + "]");
        for (size_t s = 0; s < samples; ++s) {
            volatile size_t sample = dd(gen);
        }
    }
}

void benchmark_bt_sampling(const std::vector<double>& weights, size_t samples, std::mt19937_64& gen, std::string name) {
    BinaryTree bt(weights);
    {
        tools::ScopedTimer timer("BinaryTree " + name + " [n: " + std::to_string(weights.size()) + "]");
        for (size_t s = 0; s < samples; ++s) {
            volatile size_t sample = bt.sample(gen);
        }
    }
}

int main() {
    std::random_device rd;
    size_t seed = rd();
    std::mt19937_64 gen(seed);

    std::vector<size_t> ns = {10000000, 100000000};
    size_t samples = 1000000;
    size_t repeats = 10;

    for (size_t r = 0; r < repeats; ++r) {
        for (size_t n : ns) {
            std::vector<std::pair<std::vector<double>, std::string>> weights_names = {
                    {generate_noisy_uniform_weights(n, gen), "NoisyUniform"},
                    {generate_power_law_weights(n, gen),     "PowerLaw"},
                    {generate_noisy_delta_weights(n, gen),   "NoisyDelta"}
            };
            for (auto[weights, name] : weights_names) {
                benchmark_bt_sampling(weights, samples, gen, name);
                benchmark_at_sampling(weights, samples, gen, name);
                benchmark_pa_sampling(weights, samples, gen, name);
            }
        }
    }

    return 0;
}
