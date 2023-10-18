#include <cstdint>
#include <random>
#include <sampling/ScopedTimer.hpp>
#include <sampling/DynamicProposalArray.hpp>
#include <sampling/DynamicProposalArrayStar.hpp>
#include <sampling/LogCascade.hpp>
#include <sampling/BinaryTree.hpp>

using namespace sampling;

template <typename Algo>
void benchmark_insertion(size_t nl, size_t nu, double f, size_t samples, std::string name, std::mt19937_64& gen) {
    std::uniform_real_distribution<double> weight_dist(0, nl);
    std::vector<double> weights;
    for (size_t i = 0; i < nl; ++i) weights.push_back(weight_dist(gen));
    Algo ds(weights);
    size_t n = nl;
    while (n < nu) {
        {
            tools::ScopedTimer timer(name + " Insert [n: " + std::to_string(n) + "]");
            for (size_t s = 0; s < samples; ++s) {
                volatile size_t sample = ds.sample(gen);
            }
        }
        size_t nf = n * f;
        while (n < nf) {
            double w = weight_dist(gen);
            ds.push(w);
            n++;
        }
    }
}

template <typename Algo>
void benchmark_removal(size_t nl, size_t nu, double f, size_t samples, std::string name, std::mt19937_64& gen) {
    std::uniform_real_distribution<double> weight_dist(0, nl);
    std::vector<double> weights;
    for (size_t i = 0; i < nu; ++i) weights.push_back(weight_dist(gen));
    Algo ds(weights);
    size_t n = nu;
    while (n > nl / f) {
        {
            tools::ScopedTimer timer(name + " Erase [n: " + std::to_string(n) + "]");
            for (size_t s = 0; s < samples; ++s) {
                volatile size_t sample = ds.sample(gen);
            }
        }
        size_t nf = n / f;
        while (n > nf) {
            ds.pop();
            n--;
        }
    }
}

void benchmark_insertion_bt(size_t nl, size_t nu, double f, size_t samples, std::string name, std::mt19937_64& gen) {
    std::uniform_real_distribution<double> weight_dist(0, nl);
    std::vector<double> weights(nu * f, 0.0);
    for (size_t i = 0; i < nl; ++i) weights[i] = weight_dist(gen);
    BinaryTree ds(weights);
    size_t n = nl;
    while (n < nu) {
        {
            tools::ScopedTimer timer(name + " Insert [n: " + std::to_string(n) + "]");
            for (size_t s = 0; s < samples; ++s) {
                volatile size_t sample = ds.sample(gen);
            }
        }
        size_t nf = n * f;
        while (n < nf) {
            double w = weight_dist(gen);
            ds.update(n, w);
            n++;
        }
    }
}

void benchmark_removal_bt(size_t nl, size_t nu, double f, size_t samples, std::string name, std::mt19937_64& gen) {
    std::uniform_real_distribution<double> weight_dist(0, nl);
    std::vector<double> weights;
    for (size_t i = 0; i < nu; ++i) weights.push_back(weight_dist(gen));
    BinaryTree ds(weights);
    size_t n = nu;
    while (n > nl / f) {
        {
            tools::ScopedTimer timer(name + " Erase [n: " + std::to_string(n) + "]");
            for (size_t s = 0; s < samples; ++s) {
                volatile size_t sample = ds.sample(gen);
            }
        }
        size_t nf = n / f;
        while (n > nf) {
            ds.update(n - 1, 0);
            n--;
        }
    }
}

int main() {
    std::random_device rd;
    size_t seed = rd();
    std::mt19937_64 gen(seed);

    size_t nl = (1<<16);
    size_t nu = (1<<26);
    double f = std::sqrt(std::sqrt(2.0));
    size_t samples = 1000000;
    size_t repeats = 10;

    for (size_t r = 0; r < repeats; ++r) {
        benchmark_insertion<DynamicProposalArray>(nl, nu, f, samples, "ProposalArray", gen);
        benchmark_insertion<DynamicProposalArrayStar>(nl, nu, f, samples, "ProposalArrayStar", gen);
        benchmark_insertion<LogCascade<1>>(nl, nu, f, samples, "LogCascade", gen);
        benchmark_insertion_bt(nl, nu, f, samples, "BinaryTree", gen);
        benchmark_removal<DynamicProposalArray>(nl, nu, f, samples, "ProposalArray", gen);
        benchmark_removal<DynamicProposalArrayStar>(nl, nu, f, samples, "ProposalArrayStar", gen);
        benchmark_removal<LogCascade<1>>(nl, nu, f, samples, "LogCascade", gen);
        benchmark_removal_bt(nl, nu, f, samples, "BinaryTree", gen);
    }

    return 0;
}