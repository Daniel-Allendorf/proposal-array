#pragma once
#include <cassert>
#include <random>
#include <vector>

#include <iostream>

namespace sampling {

using index_t = uint64_t;

class ProposalArray {
public:
    ProposalArray(const std::vector<double>& weights, double W) :
        weights_(weights), counts_(weights.size()), N_(weights.size()), W_(W), real_dist_(0, 1) {
        assert(N_ > 0);
        P_.reserve(2 * N_);
        max_ = 0.;
        for (size_t i = 0; i < N_; ++i) {
            counts_[i] = std::ceil((weights_[i] * N_) / W_);
            for (size_t j = 0; j < counts_[i]; ++j) {
                P_.push_back(i);
            }
            max_ = std::max(max_, weights_[i] / counts_[i]);
        }
        entry_dist_ = std::uniform_int_distribution<size_t>(0, P_.size() - 1);
    }

    ProposalArray() : N_(0) {}

    template <typename Generator>
    index_t sample(Generator&& gen) {
        assert(N_ > 0);
        do {
            index_t element = P_[entry_dist_(gen)];
            assert(weights_[element] / counts_[element] <= max_);
            double p_acc = (weights_[element] / counts_[element]) / max_;
            if (real_dist_(gen) < p_acc) {
                return element;
            }
        } while (true);
    }

private:
    std::vector<index_t> P_;
    std::vector<index_t> counts_;
    std::vector<double> weights_;
    std::uniform_int_distribution<size_t> entry_dist_;
    std::uniform_real_distribution<double> real_dist_;
    size_t N_;
    double W_;
    double max_;
};

}