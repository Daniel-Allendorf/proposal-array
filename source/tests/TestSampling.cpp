#include <cstdint>
#include <iostream>
#include <random>
#include <sampling/AliasTable.hpp>
#include <sampling/ProposalArray.hpp>
#include <sampling/DynamicProposalArray.hpp>
#include <sampling/DynamicProposalArrayStar.hpp>

using namespace sampling;

int main() {
    std::random_device rd;
    size_t seed = rd();
    std::mt19937_64 gen(seed);

    const std::vector<double> weights = {5.0, 1.5, 0.1, 2.0};
    const size_t N = weights.size();
    const double W = 8.6;
    const size_t samples = 1000000 * W;

    std::discrete_distribution dd(weights.begin(), weights.end());
    std::vector<size_t> dd_counts(N, 0);
    for (size_t s = 0; s < samples; ++s) {
        size_t i = dd(gen);
        dd_counts[i]++;
    }
    std::cout << "Discrete Distribution [";
    for (size_t i = 0; i < weights.size(); ++i) {
        std::cout << dd_counts[i];
        if (i < weights.size() - 1) std::cout << " ";
    }
    std::cout << "]" << std::endl;

    AliasTable at(weights);
    std::vector<size_t> at_counts(N, 0);
    for (size_t s = 0; s < samples; ++s) {
        size_t i = at.sample(gen);
        at_counts[i]++;
    }
    std::cout << "Alias Table [";
    for (size_t i = 0; i < weights.size(); ++i) {
        std::cout << at_counts[i];
        if (i < weights.size() - 1) std::cout << " ";
    }
    std::cout << "]" << std::endl;

    ProposalArray pa(weights);
    std::vector<size_t> pa_counts(N, 0);
    for (size_t s = 0; s < samples; ++s) {
        size_t i = pa.sample(gen);
        pa_counts[i]++;
    }
    std::cout << "Proposal Array [";
    for (size_t i = 0; i < weights.size(); ++i) {
        std::cout << pa_counts[i];
        if (i < weights.size() - 1) std::cout << " ";
    }
    std::cout << "]" << std::endl;


    const std::vector<double> mod_weights = {2.5, 10.0, 1.0, 0.01};
    const double mod_W = 13.51;
    const size_t mod_samples = 1000000 * mod_W;

    DynamicProposalArray dyn_pa(weights);
    std::vector<size_t> dyn_pa_counts(N, 0);
    std::vector<size_t> dyn_pa_mod_counts(N, 0);
    for (size_t s = 0; s < samples; ++s) {
        size_t i = dyn_pa.sample(gen);
        dyn_pa_counts[i]++;
    }
    for (size_t i = 0; i < weights.size(); ++i) {
        dyn_pa.update(i, mod_weights[i]);
    }
    for (size_t s = 0; s < mod_samples; ++s) {
        size_t i = dyn_pa.sample(gen);
        dyn_pa_mod_counts[i]++;
    }
    std::cout << "Dynamic Proposal Array [";
    for (size_t i = 0; i < weights.size(); ++i) {
        std::cout << dyn_pa_counts[i];
        if (i < weights.size() - 1) std::cout << " ";
    }
    std::cout << " | ";
    for (size_t i = 0; i < weights.size(); ++i) {
        std::cout << dyn_pa_mod_counts[i];
        if (i < weights.size() - 1) std::cout << " ";
    }
    std::cout << "]" << std::endl;

    DynamicProposalArrayStar dyn_pa_star(weights);
    std::vector<size_t> dyn_pa_star_counts(N, 0);
    std::vector<size_t> dyn_pa_star_mod_counts(N, 0);
    for (size_t s = 0; s < samples; ++s) {
        size_t i = dyn_pa_star.sample(gen);
        dyn_pa_star_counts[i]++;
    }
    for (size_t i = 0; i < weights.size(); ++i) {
        dyn_pa_star.update(i, mod_weights[i]);
    }
    for (size_t s = 0; s < mod_samples; ++s) {
        size_t i = dyn_pa_star.sample(gen);
        dyn_pa_star_mod_counts[i]++;
    }
    std::cout << "Dynamic Proposal Array Star [";
    for (size_t i = 0; i < weights.size(); ++i) {
        std::cout << dyn_pa_star_counts[i];
        if (i < weights.size() - 1) std::cout << " ";
    }
    std::cout << " | ";
    for (size_t i = 0; i < weights.size(); ++i) {
        std::cout << dyn_pa_star_mod_counts[i];
        if (i < weights.size() - 1) std::cout << " ";
    }
    std::cout << "]" << std::endl;

    return 0;
}
