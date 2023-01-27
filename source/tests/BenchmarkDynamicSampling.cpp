#include <cstdint>
#include <random>
#include <sampling/DynamicProposalArray.hpp>
#include <sampling/DynamicProposalArrayStar.hpp>
#include <sampling/ScopedTimer.hpp>

using namespace sampling;

template <typename Algo>
void benchmark_random_increase(index_t n, index_t g, std::mt19937_64& gen) {
    std::vector<double> weights(n, 1.0);
    index_t steps = 100 * n;
    index_t substeps = steps / g;
    index_t samples = 1000000;
    std::uniform_int_distribution<index_t> index_dist(0, n - 1);
    std::vector<index_t> updates;
    for (index_t t = 0; t < steps; ++t) {
        index_t i = index_dist(gen);
        updates.push_back(i);
    }
    {
        Algo pa(weights, n);
        for (index_t t = 0; t < steps;) {
            {
                incpwl::ScopedTimer timer(pa.name() + " RandomIncrease [n: " + std::to_string(t) + "]");
                for (size_t s = 0; s < samples; ++s) {
                    volatile index_t sample = pa.sample(gen);
                }
            }
            for (index_t s = 0; s < substeps; ++s) {
                index_t i = updates[t + s];
                weights[i] += 1.0;
                pa.update(i, weights[i]);
            }
            t += substeps;
        }
    }
    {
        Algo pa_reverse(weights, n + steps);
        for (index_t t = 0; t < steps;) {
            {
                incpwl::ScopedTimer timer(pa_reverse.name() + " RandomDecrease [n: " + std::to_string(t) + "]");
                for (size_t s = 0; s < samples; ++s) {
                    volatile index_t sample = pa_reverse.sample(gen);
                }
            }
            for (index_t s = 0; s < substeps; ++s) {
                index_t i = updates[steps - t - s - 1];
                weights[i] -= 1.0;
                pa_reverse.update(i, weights[i]);
            }
            t += substeps;
        }
    }
}

template <typename Algo>
void benchmark_polya_urn(index_t n, index_t g, std::mt19937_64& gen) {
    std::vector<double> weights(n, 1.0);
    index_t steps = 100 * n;
    index_t substeps = steps / g;
    index_t samples = 1000000;
    std::vector<index_t> updates;
    {
        std::vector<double> auxilary_weights(n, 1.0);
        Algo auxilary_pa(auxilary_weights, n);
        for (index_t t = 0; t < steps; ++t) {
            index_t i = auxilary_pa.sample(gen);
            auxilary_weights[i] += 1.0;
            auxilary_pa.update(i, auxilary_weights[i]);
            updates.push_back(i);
        }
    }
    {
        Algo pa(weights, n);
        for (index_t t = 0; t < steps;) {
            {
                incpwl::ScopedTimer timer(pa.name() + " PolyaUrn [n: " + std::to_string(t) + "]");
                for (size_t s = 0; s < samples; ++s) {
                    volatile index_t sample = pa.sample(gen);
                }
            }
            for (index_t s = 0; s < substeps; ++s) {
                index_t i = updates[t + s];
                weights[i] += 1.0;
                pa.update(i, weights[i]);
            }
            t += substeps;
        }
    }
    {
        Algo pa_reverse(weights, n + steps);
        for (index_t t = 0; t < steps;) {
            {
                incpwl::ScopedTimer timer(pa_reverse.name() + " PolyaUrnReverse [n: " + std::to_string(t) + "]");
                for (size_t s = 0; s < samples; ++s) {
                    volatile index_t sample = pa_reverse.sample(gen);
                }
            }
            for (index_t s = 0; s < substeps; ++s) {
                index_t i = updates[steps - t - s - 1];
                weights[i] -= 1.0;
                pa_reverse.update(i, weights[i]);
            }
            t += substeps;
        }
    }
}

template <typename Algo>
void benchmark_single_increase(index_t n, index_t g, std::mt19937_64& gen) {
    std::vector<double> weights(n, 1.0);
    index_t steps = 100 * n;
    index_t substeps = steps / g;
    index_t samples = 1000000;
    {
        Algo pa(weights, n);
        for (index_t t = 0; t < steps;) {
            {
                incpwl::ScopedTimer timer(pa.name() + " SingleIncrease [n: " + std::to_string(t) + "]");
                for (size_t s = 0; s < samples; ++s) {
                    volatile index_t sample = pa.sample(gen);
                }
            }
            for (index_t s = 0; s < substeps; ++s) {
                index_t i = 0;
                weights[i] += 1.0;
                pa.update(i, weights[i]);
            }
            t += substeps;
        }
    }
    {
        Algo pa_reverse(weights, n + steps);
        for (index_t t = 0; t < steps;) {
            {
                incpwl::ScopedTimer timer(pa_reverse.name() + " SingleDecrease [n: " + std::to_string(t) + "]");
                for (size_t s = 0; s < samples; ++s) {
                    volatile index_t sample = pa_reverse.sample(gen);
                }
            }
            for (index_t s = 0; s < substeps; ++s) {
                index_t i = 0;
                weights[i] -= 1.0;
                pa_reverse.update(i, weights[i]);
            }
            t += substeps;
        }
    }
}

int main() {
    std::random_device rd;
    size_t seed = rd();
    std::mt19937_64 gen(seed);

    index_t n = 10000000;
    index_t g = 100;
    const size_t repeats = 10;

    for (size_t r = 0; r < repeats; ++r) {
        benchmark_random_increase<DynamicProposalArray>(n, g, gen);
        benchmark_random_increase<DynamicProposalArrayStar>(n, g, gen);
        benchmark_polya_urn<DynamicProposalArray>(n, g, gen);
        benchmark_polya_urn<DynamicProposalArrayStar>(n, g, gen);
        benchmark_single_increase<DynamicProposalArray>(n, g, gen);
        benchmark_single_increase<DynamicProposalArrayStar>(n, g, gen);
    }

    return 0;
}