#pragma once
#include <cassert>
#include <random>
#include <vector>

namespace sampling {

class BinaryTree {
    constexpr static size_t K_ = 2; // number of children per node, e.g. K = 2 corresponds to a binary tree
public:
    BinaryTree(const std::vector<double>& weights) : real_dist_(0, 1) {
        assert(weights.size() > 0);
        N_ = weights.size();
        L_ = std::ceil(std::log(weights.size()) / std::log(K_)); // number of layers
        S_ = std::pow(K_, L_); // offset for leafs
        T_ = std::vector<double>(S_ * K_, 0.0); // T_[0] is empty so that we don't need to subtract 1 from indices
        // initialize leafs
        for (size_t i = 0; i < N_; ++i) {
            T_[S_ + i] = weights[i];
        }
        // initialize inner nodes
        for (size_t j = S_ - 1; j > 0; --j) {
            for (size_t k = 0; k < K_; ++k) {
                T_[j] += T_[K_ * j + k];
            }
        }
    }

    template <typename Generator>
    size_t sample(Generator&& gen) {
        // sample x from U(0, C), C = sum_i w_i
        auto x = T_[1] * real_dist_(gen);
        // find corresponding leaf by branching at inner nodes
        size_t i = 1;
        while (i < S_) {
            for (size_t k = 0; k < K_; ++k) {
                double wk = T_[K_ * i + k];
                if (x < wk) {
                    i = K_ * i + k;
                    break;
                }
                x -= wk;
            }
        }
        return i - S_;
    }

    void update(size_t i, double w) {
        size_t j = S_ + i;
        assert(j < S_ * K_);
        double dw = w - T_[j];
        // update leaf, then all parents
        while (j > 0) {
            T_[j] += dw;
            j /= K_;
        }
    }

private:
    std::vector<double> T_;
    std::uniform_real_distribution<double> real_dist_;
    size_t N_;
    size_t L_;
    size_t S_;
};

}