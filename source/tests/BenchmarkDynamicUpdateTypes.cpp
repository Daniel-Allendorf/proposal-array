#include <cstdint>
#include <random>
#include <sampling/DynamicProposalArray.hpp>
#include <sampling/DynamicProposalArrayStar.hpp>
#include <sampling/ScopedTimer.hpp>

using namespace sampling;

template <typename Algo>
void benchmark_update_types(index_t n, size_t repeats, std::mt19937_64& gen) {
    std::vector<double> weights(n, 1.0);
    Algo pa(weights, n);
    std::uniform_int_distribution<index_t> index_dist(0, n - 1);
    std::vector<std::pair<index_t, double>> updates;
    updates.emplace_back(n - 1, n / 2);
    updates.emplace_back(n - 1, 0);
    updates.emplace_back(n - 1, n / 2);
    updates.emplace_back(n - 1, 1);
    std::vector<index_t> t_remap = {0, 3, 2, 1};
    for (size_t r = 0; r < repeats; ++r) {
        for (index_t t = 0; t < updates.size(); ++t) {
            incpwl::ScopedTimer timer(pa.name() + " Types [n: " + std::to_string(t_remap[t]) + "]");
            auto[i, w] = updates[t];
            pa.update(i, w);
        }
    }
}

int main() {
    std::random_device rd;
    size_t seed = rd();
    std::mt19937_64 gen(seed);

    index_t n = 100000000;
    const size_t repeats = 100;

    benchmark_update_types<DynamicProposalArray>(n, repeats, gen);

    return 0;
}
