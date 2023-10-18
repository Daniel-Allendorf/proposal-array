#include <cstdint>
#include <numbers>
#include <random>
#include <sampling/ScopedTimer.hpp>
#include <sampling/AliasTable.hpp>
#include <sampling/ProposalArray.hpp>

using namespace sampling;

std::vector<double> generate_noisy_uniform_weights(size_t n, std::mt19937_64& gen) {
    std::uniform_real_distribution<double> weight_dist(1, n);
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
    std::uniform_real_distribution<double> weight_dist(1, n);
    double weight_sum;
    std::vector<double> weights;
    weights.reserve(n);
    for (size_t i = 0; i < n - 1; ++i) {
        double random_weight = weight_dist(gen);
        weight_sum += random_weight;
        weights.push_back(random_weight);
    }
    weights.push_back(weight_sum);
    return weights;
}

void benchmark_at_construction(const std::vector<double>& weights, std::string name) {
    tools::ScopedTimer timer("AliasTable " + name + " [n: " + std::to_string(weights.size()) + "]");
    AliasTable at(weights);
}

void benchmark_pa_construction(const std::vector<double>& weights, std::string name) {
    tools::ScopedTimer timer("ProposalArray " + name + " [n: " + std::to_string(weights.size()) + "]");
    ProposalArray pa(weights);
}

void benchmark_dd_construction(const std::vector<double>& weights, std::string name) {
    tools::ScopedTimer timer("DiscreteDistribution " + name + " [n: " + std::to_string(weights.size()) + "]");
    std::discrete_distribution dd(weights.begin(), weights.end());
}

int main() {
    std::random_device rd;
    size_t seed = rd();
    std::mt19937_64 gen(seed);

    std::vector<size_t> ns = {100000, 1000000, 10000000, 100000000};
    size_t repeats = 1;

    for (auto n : ns) {
        for (size_t r = 0; r < repeats; ++r) {
            std::vector<std::pair<std::vector<double>, std::string>> weights_names = {
                    { generate_noisy_uniform_weights(n, gen), "NoisyUniform" },
                    { generate_power_law_weights(n, gen), "PowerLaw" },
                    { generate_noisy_delta_weights(n, gen), "NoisyDelta" }
            };
            for (auto [weights, name] : weights_names) {
                benchmark_at_construction(weights, name);
                benchmark_pa_construction(weights, name);
            }
        }
    }

    return 0;
}
