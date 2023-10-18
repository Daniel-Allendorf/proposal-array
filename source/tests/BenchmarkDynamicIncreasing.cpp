#include <cstdint>
#include <random>
#include <sampling/ScopedTimer.hpp>
#include <sampling/DynamicProposalArray.hpp>
#include <sampling/DynamicProposalArrayStar.hpp>

using namespace sampling;

template <typename Algo>
void benchmark_increasing(size_t n, std::string name, std::mt19937_64& gen) {
    std::vector<double> weights(n, 1.0);
    std::uniform_int_distribution<size_t> index_dist(0, n - 1);
    std::vector<std::pair<size_t, size_t>> updates;
    for (size_t t = 1; t <= n; t *= 2) {
        size_t i = index_dist(gen);
        updates.emplace_back(i, t);
    }
    for (auto [i, dw] : updates) {
        Algo pa(weights);
        {
            tools::ScopedTimer timer(name + " Increasing [n: " + std::to_string(dw) + "]");
            pa.update(i, dw);
        }
    }
}

int main() {
    std::random_device rd;
    size_t seed = rd();
    std::mt19937_64 gen(seed);

    size_t n = 10000000;
    size_t repeats = 10;

    for (size_t r = 0; r < repeats; ++r) {
        benchmark_increasing<DynamicProposalArray>(n, "ProposalArray", gen);
        benchmark_increasing<DynamicProposalArrayStar>(n, "ProposalArrayStar", gen);
    }

    return 0;
}
