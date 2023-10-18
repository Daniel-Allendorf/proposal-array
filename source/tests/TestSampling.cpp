#include <iostream>
#include <random>
#include <sampling/AliasTable.hpp>
#include <sampling/ProposalArray.hpp>
#include <sampling/DynamicProposalArray.hpp>
#include <sampling/DynamicProposalArrayStar.hpp>
#include <sampling/BinaryTree.hpp>
#include <sampling/LogCascade.hpp>

using namespace sampling;

template <typename Algo, typename Generator>
void test_ds(const std::vector<double>& weights, size_t samples, const char* name, Generator&& gen) {
    Algo ds(weights);
    std::vector<size_t> counts(weights.size(), 0);
    for (size_t s = 0; s < samples; ++s) {
        size_t i = ds.sample(gen);
        counts[i]++;
    }
    std::cout << name << " [";
    for (size_t i = 0; i < weights.size(); ++i) {
        std::cout << counts[i];
        if (i < weights.size() - 1) std::cout << " ";
    }
    std::cout << "]" << std::endl;
}

template <typename Algo, typename Generator>
void test_dynamic_ds(const std::vector<double>& weights, const std::vector<double>& mod_weights,
                     size_t samples, size_t mod_samples, const char* name, Generator&& gen) {
    Algo ds(weights);
    std::vector<size_t> counts(weights.size(), 0);
    std::vector<size_t> mod_counts(mod_weights.size(), 0);
    for (size_t s = 0; s < samples; ++s) {
        size_t i = ds.sample(gen);
        counts[i]++;
    }
    for (size_t i = 0; i < weights.size(); ++i) {
        ds.update(i, mod_weights[i]);
    }
    for (size_t s = 0; s < mod_samples; ++s) {
        size_t i = ds.sample(gen);
        mod_counts[i]++;
    }
    std::cout << name << " [";
    for (size_t i = 0; i < weights.size(); ++i) {
        std::cout << counts[i];
        if (i < weights.size() - 1) std::cout << " ";
    }
    std::cout << " | ";
    for (size_t i = 0; i < weights.size(); ++i) {
        std::cout << mod_counts[i];
        if (i < weights.size() - 1) std::cout << " ";
    }
    std::cout << "]" << std::endl;
}

int main() {
    std::random_device rd;
    size_t seed = rd();
    std::mt19937_64 gen(seed);

    const std::vector<double> weights = {5.0, 1.5, 0.1, 2.0};
    const double W = 8.6;
    const size_t samples = 1000000 * W;

    test_ds<AliasTable>(weights, samples, "Alias Table", gen);
    test_ds<ProposalArray>(weights, samples, "Proposal Array", gen);

    const std::vector<double> mod_weights = {2.5, 10.0, 1.0, 0.01};
    const double mod_W = 13.51;
    const size_t mod_samples = 1000000 * mod_W;

    test_dynamic_ds<DynamicProposalArray>(weights, mod_weights, samples, mod_samples, "Dynamic PA", gen);
    test_dynamic_ds<DynamicProposalArrayStar>(weights, mod_weights, samples, mod_samples, "Dynamic PA*", gen);
    test_dynamic_ds<BinaryTree>(weights, mod_weights, samples, mod_samples, "Binary Tree", gen);
    test_dynamic_ds<LogCascade<3>>(weights, mod_weights, samples, mod_samples, "Log Cascade Iterated", gen);

    return 0;
}
