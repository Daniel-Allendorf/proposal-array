#include <cstdint>
#include <numbers>
#include <random>
#include <sampling/AliasTable.hpp>
#include <sampling/ProposalArray.hpp>
#include <sampling/ScopedTimer.hpp>

using namespace sampling;

std::vector<double> generate_noisy_uniform_weights(index_t n, std::mt19937_64& gen) {
    std::uniform_real_distribution<double> weight_dist(0, 1);
    std::vector<double> weights;
    weights.reserve(n);
    for (index_t i = 0; i < n; ++i) {
        double random_weight = weight_dist(gen);
        weights.push_back(random_weight);
    }
    return weights;
}

std::vector<double> generate_power_law_weights(index_t n, std::mt19937_64& gen) {
    std::uniform_real_distribution<double> real_dist(0, 1);
    std::vector<double> weights;
    weights.reserve(n);
    for (index_t i = 0; i < n; ++i) {
        double C = std::numbers::pi * std::numbers::pi / 6;
        index_t w = 1;
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

std::vector<double> generate_noisy_delta_weights(index_t n, std::mt19937_64& gen) {
    std::uniform_real_distribution<double> weight_dist(0, 1);
    double weight_sum;
    std::vector<double> weights;
    weights.reserve(n);
    for (index_t i = 0; i < n - 1; ++i) {
        double random_weight = weight_dist(gen);
        weight_sum += random_weight;
        weights.push_back(random_weight);
    }
    weights.push_back(weight_sum);
    return weights;
}

void benchmark_at_construction(const std::vector<double>& weights, std::string name) {
    index_t n = weights.size();
    double W = 0.;
    for (auto weight : weights) {
        W += weight;
    }
    AliasTable at;
    {
        incpwl::ScopedTimer timer("AliasTable " + name + " [n: " + std::to_string(n) + "]");
        at = AliasTable(weights, W);
    }
}

void benchmark_pa_construction(const std::vector<double>& weights, std::string name) {
    index_t n = weights.size();
    double W = 0.;
    for (auto weight : weights) {
        W += weight;
    }
    ProposalArray pa;
    {
        incpwl::ScopedTimer timer("ProposalArray " + name + " [n: " + std::to_string(n) + "]");
        pa = ProposalArray(weights, W);
    }
}

int main() {
    std::random_device rd;
    size_t seed = rd();
    std::mt19937_64 gen(seed);

    const std::vector<index_t> ns = {1000000, 10000000, 100000000, 1000000000};
    const size_t repeats = 1;

    for (const auto n : ns) {
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
