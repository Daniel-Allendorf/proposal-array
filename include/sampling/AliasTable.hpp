#pragma once
#include <cassert>
#include <random>
#include <vector>
#include <tuple>

namespace sampling {

class AliasTable {
public:
    AliasTable(const std::vector<double>& weights) :
        table_(weights.size()), entry_dist_(0, weights.size() - 1), real_dist_(0, 1) {
        assert(weights.size() > 0);
        size_t N = weights.size();
        double W = 0;
        for (auto w : weights) W += w;
        std::vector<size_t> overfull;
        std::vector<size_t> underfull;
        overfull.reserve(N);
        underfull.reserve(N);
        for (size_t i = 0; i < N; ++i) {
            double p_i = weights[i] / W;
            size_t element = i;
            size_t alias = N;
            double threshold = N * p_i;
            if (threshold < 1.0) {
                underfull.push_back(i);
            } else if (threshold > 1.0) {
                overfull.push_back(i);
            } else {
                threshold = 1.0;
                alias = i;
            }
            table_[i] = std::make_tuple(element, alias, threshold);
        }
        while (!underfull.empty() || !overfull.empty()) {
            if (overfull.empty()) {
                size_t i = underfull.back();
                underfull.pop_back();
                table_[i] = std::make_tuple(i, i, 1.0);
                continue;
            }
            if (underfull.empty()) {
                size_t j = overfull.back();
                overfull.pop_back();
                table_[j] = std::make_tuple(j, j, 1.0);
                continue;
            }
            size_t i = underfull.back();
            size_t j = overfull.back();
            underfull.pop_back();
            overfull.pop_back();
            auto [element_i, alias_i, threshold_i] = table_[i];
            auto [element_j, alias_j, threshold_j] = table_[j];
            double threshold = threshold_i + threshold_j - 1.0;
            if (threshold < 1.0) {
                underfull.push_back(j);
            } else if (threshold > 1.0) {
                overfull.push_back(j);
            } else {
                threshold = 1.0;
                alias_j = j;
            }
            table_[i] = std::make_tuple(element_i, j, threshold_i);
            table_[j] = std::make_tuple(element_j, alias_j, threshold);
        }
    }

    template <typename Generator>
    size_t sample(Generator&& gen) {
        assert(N_ > 0);
        size_t i = entry_dist_(gen);
        auto [element, alias, threshold] = table_[i];
        return real_dist_(gen) < threshold ? element : alias;
    }

private:
    std::vector<std::tuple<size_t, size_t, double>> table_;
    std::uniform_int_distribution<size_t> entry_dist_;
    std::uniform_real_distribution<double> real_dist_;
};

}