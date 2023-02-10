#pragma once
#include <cassert>
#include <random>
#include <vector>

namespace sampling {

class ProposalArray {
public:
    ProposalArray(const std::vector<double>& weights) : real_dist_(0, 1) {
        assert(weights.size() > 0);
        double W = 0;
        for (auto w : weights) W += w;
        size_t N = weights.size();
        double avg = W / N;
        P_.reserve(N);
        R_.resize(N);
        for (size_t i = 0; i < N; ++i) {
            double weight = weights[i];
            size_t count = std::floor(weight / avg);
            for (size_t j = 0; j < count; ++j) {
                P_.push_back(i);
            }
            R_[i] = (weight / avg) - count;
        }
        entry_dist_ = std::uniform_int_distribution<size_t>(0, R_.size() + P_.size() - 1);
    }

    template <typename Generator>
    size_t sample(Generator&& gen) {
        do {
            auto i = entry_dist_(gen);
            if (i < R_.size()) {
                double p_acc = R_[i];
                if (real_dist_(gen) < p_acc) {
                    return i;
                }
            } else {
                return P_[i - R_.size()];
            }
        } while (true);
    }

private:
    std::vector<double> R_;
    std::vector<size_t> P_;
    std::uniform_int_distribution<size_t> entry_dist_;
    std::uniform_real_distribution<double> real_dist_;
};

}