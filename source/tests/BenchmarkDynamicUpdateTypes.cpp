#include <cstdint>
#include <random>
#include <sampling/DynamicProposalArray.hpp>
#include <sampling/DynamicProposalArrayStar.hpp>
#include <sampling/ScopedTimer.hpp>

using namespace sampling;

template <typename Algo>
void benchmark_update_types(size_t n, std::string name, std::mt19937_64& gen) {
    std::vector<double> weights(n, 1.0);
    std::uniform_int_distribution<size_t> index_dist(0, n - 1);
    size_t steps = 10;
    for (size_t s = 0; s < steps; ++s) {
        size_t i = index_dist(gen);
        {
            Algo pa(weights);
            weights[i] = 2.0;
            {
                incpwl::ScopedTimer timer(name + " Types [n: 0]");
                pa.update(i, weights[i]);
            }
        }
        {
            Algo pa(weights);
            weights[i] = 1.0;
            {
                incpwl::ScopedTimer timer(name + " Types [n: 3]");
                pa.update(i, weights[i]);
            }
        }
    }
}

int main() {
    std::random_device rd;
    size_t seed = rd();
    std::mt19937_64 gen(seed);

    size_t n = 10000000;

    benchmark_update_types<DynamicProposalArray>(n, "ProposalArray", gen);
    benchmark_update_types<DynamicProposalArrayStar>(n, "ProposalArrayStar", gen);

    return 0;
}
