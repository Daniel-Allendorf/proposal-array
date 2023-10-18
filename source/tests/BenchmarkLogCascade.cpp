#include <cstdint>
#include <random>
#include <sampling/ScopedTimer.hpp>
#include <sampling/LogCascade.hpp>

using namespace sampling;

std::vector<double> generate_noisy_uniform_weights(size_t n, std::mt19937_64& gen) {
    std::uniform_real_distribution<double> weight_dist(0, n);
    std::vector<double> weights;
    weights.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        double random_weight = weight_dist(gen);
        weights.push_back(random_weight);
    }
    return weights;
}

template <typename Algo>
void benchmark_sampling(const std::vector<double>& weights, size_t samples, std::string name, std::mt19937_64& gen) {
    Algo lc(weights);
    {
        tools::ScopedTimer timer(name + " NoisyUniform" + " [n: " + std::to_string(weights.size()) + "]");
        for (size_t s = 0; s < samples; ++s) {
            volatile size_t sample = lc.sample(gen);
        }
    }
}

int main() {
    std::random_device rd;
    size_t seed = rd();
    std::mt19937_64 gen(seed);

    size_t samples = 1000000;
    size_t repeats = 10;

    for (size_t r = 0; r < repeats; ++r) {
        for (size_t n = (2<<19); n <= (2<<26); n *= 2) {
            auto weights = generate_noisy_uniform_weights(n, gen);
            benchmark_sampling<LogCascade<1>>(weights, samples, "LogCascade1L", gen);
            benchmark_sampling<LogCascade<2>>(weights, samples, "LogCascade2L", gen);
            benchmark_sampling<LogCascade<3>>(weights, samples, "LogCascade3L", gen);
            benchmark_sampling<LogCascade<4>>(weights, samples, "LogCascade4L", gen);
        }
    }

    return 0;
}