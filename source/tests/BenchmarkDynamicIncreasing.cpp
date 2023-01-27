#include <cstdint>
#include <random>
#include <sampling/DynamicProposalArray.hpp>
#include <sampling/DynamicProposalArrayStar.hpp>
#include <sampling/ScopedTimer.hpp>

using namespace sampling;

template <typename Algo>
void benchmark_increasing(index_t n, std::mt19937_64& gen) {
    std::vector<double> weights(n, 1.0);
    Algo pa(weights, n);
    std::uniform_int_distribution<index_t> index_dist(0, n - 1);
    std::vector<std::pair<index_t, index_t>> updates;
    for (index_t t = 1; t <= n * n; t *= 2) {
        index_t i = index_dist(gen);
        updates.emplace_back(i, t);
    }
    for (auto [i, dw] : updates) {
        incpwl::ScopedTimer timer(pa.name() + " Increasing [n: " + std::to_string(dw) + "]");
        pa.update(i, dw);
    }
}

int main() {
    std::random_device rd;
    size_t seed = rd();
    std::mt19937_64 gen(seed);

    index_t n = 100000000;
    const size_t repeats = 5;

    for (size_t r = 0; r < repeats; ++r) {
        benchmark_increasing<DynamicProposalArray>(n, gen);
        benchmark_increasing<DynamicProposalArrayStar>(n, gen);
    }

    return 0;
}
