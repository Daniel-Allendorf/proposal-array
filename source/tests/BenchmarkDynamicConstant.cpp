#include <cstdint>
#include <random>
#include <sampling/DynamicProposalArray.hpp>
#include <sampling/DynamicProposalArrayStar.hpp>
#include <sampling/ScopedTimer.hpp>

using namespace sampling;

template <typename Algo>
void benchmark_constant(index_t n, std::mt19937_64& gen) {
    double W = n;
    std::vector<double> weights(n, 1.0);
    Algo pa(weights, W);
    std::uniform_int_distribution<index_t> index_dist(0, n - 1);
    std::vector<std::pair<index_t, double>> updates;
    index_t steps = 400;
    for (index_t t = 0; t < steps; ++t) {
        index_t i = index_dist(gen);
        weights[i] += (n / 100) * W / n;
        W +=  (n / 100) * W / n;
        updates.emplace_back(i, weights[i]);
    }
    for (index_t t = 0; t < steps; ++t) {
        incpwl::ScopedTimer timer(pa.name() + " Constant [n: " + std::to_string(t) + "]");
        auto[i, w] = updates[t];
        pa.update(i, w);
    }
}

int main() {
    std::random_device rd;
    size_t seed = rd();
    std::mt19937_64 gen(seed);

    index_t n = 100000000;
    const size_t repeats = 10;

    for (size_t r = 0; r < repeats; ++r) {
        benchmark_constant<DynamicProposalArray>(n, gen);
        benchmark_constant<DynamicProposalArrayStar>(n, gen);
    }

    return 0;
}
