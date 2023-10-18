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
        double W = std::accumulate(weights.begin(), weights.end(), 0.0);
        std::vector<size_t> hi(N); size_t hi_size = 0;
        std::vector<size_t> lo(N); size_t lo_size = 0;
        for (size_t i = 0; i < N; ++i) {
            double t = N * (weights[i] / W);
            if (t < 1.0) {
                lo[lo_size] = i; lo_size++;
            } else if (t > 1.0) {
                hi[hi_size] = i; hi_size++;
            }
            table_[i] = std::make_tuple(i, i, t);
        }
        while (lo_size > 0 && hi_size > 0) {
            size_t i = lo[lo_size - 1]; lo_size--;
            size_t j = hi[hi_size - 1]; hi_size--;
            double t_i = std::get<2>(table_[i]);
            double t_j = std::get<2>(table_[j]);
            t_j += t_i - 1.0;
            if (t_j < 1.0) {
                lo[lo_size] = j; lo_size++;
            } else if (t_j > 1.0) {
                hi[hi_size] = j; hi_size++;
            }
            table_[i] = std::make_tuple(i, j, t_i);
            table_[j] = std::make_tuple(j, j, t_j);
        }
        while (lo_size > 0) {
            size_t i = lo[lo_size - 1]; lo_size--;
            table_[i] = std::make_tuple(i, i, 1.0);
        }
        while (hi_size > 0) {
            size_t j = hi[hi_size - 1]; hi_size--;
            table_[j] = std::make_tuple(j, j, 1.0);
        }
    }

    template <typename Generator>
    size_t sample(Generator&& gen) {
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