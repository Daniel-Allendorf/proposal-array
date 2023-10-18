#include <cstdint>
#include <random>
#include <sampling/ScopedTimer.hpp>
#include <sampling/DynamicProposalArray.hpp>
#include <sampling/DynamicProposalArrayStar.hpp>
#include <sampling/LogCascade.hpp>
#include <sampling/BinaryTree.hpp>

using namespace sampling;

template <typename Algo>
void benchmark_random_increase(size_t n, size_t g, size_t samples, std::string name, std::mt19937_64& gen) {
    std::uniform_real_distribution<double> weight_dist(0, n);
    std::vector<double> weights;
    for (size_t i = 0; i < n; ++i) weights.push_back(weight_dist(gen));
    size_t steps = 100 * n;
    size_t substeps = steps / g;
    std::uniform_int_distribution<size_t> index_dist(0, n - 1);
    Algo pa(weights);
    for (size_t t = 0; t < steps;) {
        {
            tools::ScopedTimer timer(name + " RandomIncrease [n: " + std::to_string(t) + "]");
            for (size_t s = 0; s < samples; ++s) {
                volatile size_t sample = pa.sample(gen);
            }
        }
        for (size_t s = 0; s < substeps; ++s) {
            size_t i = index_dist(gen);
            weights[i] += weight_dist(gen);
            pa.update(i, weights[i]);
        }
        t += substeps;
    }
}

template <typename Algo>
void benchmark_polya_urn(size_t n, size_t g, size_t samples, std::string name, std::mt19937_64& gen) {
    std::uniform_real_distribution<double> weight_dist(0, n);
    std::vector<double> weights;
    for (size_t i = 0; i < n; ++i) weights.push_back(weight_dist(gen));
    size_t steps = 100 * n;
    size_t substeps = steps / g;
    Algo pa(weights);
    for (size_t t = 0; t < steps;) {
        {
            tools::ScopedTimer timer(name + " PolyaUrn [n: " + std::to_string(t) + "]");
            for (size_t s = 0; s < samples; ++s) {
                volatile size_t sample = pa.sample(gen);
            }
        }
        for (size_t s = 0; s < substeps; ++s) {
            size_t i = pa.sample(gen);
            weights[i] += weight_dist(gen);
            pa.update(i, weights[i]);
        }
        t += substeps;
    }
}

template <typename Algo>
void benchmark_single_increase(size_t n, size_t g, size_t samples, std::string name, std::mt19937_64& gen) {
    std::uniform_real_distribution<double> weight_dist(0, n);
    std::vector<double> weights;
    for (size_t i = 0; i < n; ++i) weights.push_back(weight_dist(gen));
    size_t steps = 100 * n;
    size_t substeps = steps / g;
    Algo pa(weights);
    for (size_t t = 0; t < steps;) {
        {
            tools::ScopedTimer timer(name + " SingleIncrease [n: " + std::to_string(t) + "]");
            for (size_t s = 0; s < samples; ++s) {
                volatile size_t sample = pa.sample(gen);
            }
        }
        for (size_t s = 0; s < substeps; ++s) {
            size_t i = 0;
            weights[i] += weight_dist(gen);
            pa.update(i, weights[i]);
        }
        t += substeps;
    }
}

int main() {
    std::random_device rd;
    size_t seed = rd();
    std::mt19937_64 gen(seed);

    size_t n = 10000000;
    size_t g = 100;
    size_t samples = 1000000;
    size_t repeats = 10;

    for (size_t r = 0; r < repeats; ++r) {
        benchmark_random_increase<DynamicProposalArray>(n, g, samples, "ProposalArray", gen);
        benchmark_random_increase<DynamicProposalArrayStar>(n, g, samples, "ProposalArrayStar", gen);
        benchmark_random_increase<LogCascade<1>>(n, g, samples, "LogCascade", gen);
        benchmark_random_increase<BinaryTree>(n, g, samples, "BinaryTree", gen);
        benchmark_polya_urn<DynamicProposalArray>(n, g, samples, "ProposalArray", gen);
        benchmark_polya_urn<DynamicProposalArrayStar>(n, g, samples, "ProposalArrayStar", gen);
        benchmark_polya_urn<LogCascade<1>>(n, g, samples, "LogCascade", gen);
        benchmark_polya_urn<BinaryTree>(n, g, samples, "BinaryTree", gen);
        benchmark_single_increase<DynamicProposalArray>(n, g, samples, "ProposalArray", gen);
        benchmark_single_increase<DynamicProposalArrayStar>(n, g, samples, "ProposalArrayStar", gen);
        benchmark_single_increase<LogCascade<1>>(n, g, samples, "LogCascade", gen);
        benchmark_single_increase<BinaryTree>(n, g, samples, "BinaryTree", gen);
    }

    return 0;
}