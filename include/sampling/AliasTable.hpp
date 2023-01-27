#pragma once
#include <cassert>
#include <random>
#include <vector>

namespace sampling {

using index_t = uint64_t;

class AliasTable {
public:
    AliasTable(const std::vector<double>& weights, double W) :
            elements_(weights.size()), thresholds_(weights.size()), aliases_(weights.size()),
            N_(weights.size()), entry_dist_(0, weights.size() - 1), real_dist_(0, 1) {
        assert(N_ > 0);
        std::vector<size_t> overfull;
        std::vector<size_t> underfull;
        overfull.reserve(N_);
        underfull.reserve(N_);
        for (size_t i = 0; i < N_; ++i) {
            double p_i = weights[i] / W;
            thresholds_[i] = N_ * p_i;
            if (thresholds_[i] < 0.9999999) {
                underfull.push_back(i);
            } else if (thresholds_[i] > 1.00000001) {
                overfull.push_back(i);
            } else {
                thresholds_[i] = 1;
                aliases_[i] = i;
            }
            elements_[i] = i;
        }
        while (!underfull.empty() || !overfull.empty()) {
            if (overfull.empty()) {
                size_t i = underfull.back();
                underfull.pop_back();
                thresholds_[i] = 1;
                aliases_[i] = i;
                continue;
            }
            if (underfull.empty()) {
                size_t j = overfull.back();
                overfull.pop_back();
                thresholds_[j] = 1;
                aliases_[j] = j;
                continue;
            }
            size_t i = underfull.back();
            size_t j = overfull.back();
            underfull.pop_back();
            overfull.pop_back();
            aliases_[i] = j;
            thresholds_[j] = thresholds_[i] + thresholds_[j] - 1;
            if (thresholds_[j] < 0.9999999) {
                underfull.push_back(j);
            } else if (thresholds_[j] > 1.00000001) {
                overfull.push_back(j);
            }
        }
    }

    AliasTable() : N_(0) {}

    template <typename Generator>
    index_t sample(Generator&& gen) {
        assert(N_ > 0);
        size_t i = entry_dist_(gen);
        return real_dist_(gen) < thresholds_[i] ? elements_[i] : aliases_[i];
    }

private:
    std::vector<index_t> elements_;
    std::vector<double> thresholds_;
    std::vector<index_t> aliases_;
    std::uniform_int_distribution<size_t> entry_dist_;
    std::uniform_real_distribution<double> real_dist_;
    size_t N_;
};

}