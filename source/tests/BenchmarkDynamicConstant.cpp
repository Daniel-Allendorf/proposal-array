#include <cstdint>
#include <random>
#include <sampling/ScopedTimer.hpp>
#include <sampling/DynamicProposalArray.hpp>
#include <sampling/DynamicProposalArrayStar.hpp>

using namespace sampling;

template <typename Algo>
void benchmark_constant(size_t n, std::string name, std::mt19937_64& gen) {
    double W = n;
    std::vector<double> weights(n, 1.0);
    Algo pa(weights);
    std::uniform_int_distribution<size_t> index_dist(0, n - 1);
    std::vector<std::pair<size_t, double>> updates;
    size_t steps = 400;
    for (size_t t = 0; t < steps; ++t) {
        size_t i = index_dist(gen);
        weights[i] += (n / 100) * W / n;
        W +=  (n / 100) * W / n;
        updates.emplace_back(i, weights[i]);
    }
    for (size_t t = 0; t < steps; ++t) {
        tools::ScopedTimer timer(name + " Constant [n: " + std::to_string(t) + "]");
        auto[i, w] = updates[t];
        pa.update(i, w);
    }
}

int main() {
    std::random_device rd;
    size_t seed = rd();
    std::mt19937_64 gen(seed);

    size_t n = 10000000;
    size_t repeats = 25;

    for (size_t r = 0; r < repeats; ++r) {
        benchmark_constant<DynamicProposalArray>(n, "ProposalArray", gen);
        benchmark_constant<DynamicProposalArrayStar>(n, "ProposalArrayStar", gen);
    }

    return 0;
}
