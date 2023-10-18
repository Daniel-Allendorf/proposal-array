#pragma once
#include <array>
#include <cassert>
#include <functional>
#include <random>
#include <vector>

namespace sampling {

template <size_t K> // number of layers
class LogCascade {
    constexpr static size_t alpha = 3; // weights must lie in [0, n^alpha]
public:
    LogCascade(const std::vector<double>& weights) : real_dist_(0, 1) {
        assert(weights.size() > 0);
        m_ = std::ceil(2 * std::log2(weights.size())) + std::ceil(std::log2(std::pow(weights.size(), alpha))) + 1;
        o_ = std::ceil(2 * std::log2(weights.size()));
        W_ = std::accumulate(weights.begin(), weights.end(), 0.0);
        // initialize weights in bottom layer
        C_[K].weights = weights;
        // initialize cascade
        for (size_t l = K; l > 0; --l) {
            C_[l].P.resize(m_);
            C_[l-1].weights = std::vector<double>(m_, 0.0);
            // distribute indices into partitions by weights
            for (size_t i = 0; i < C_[l].weights.size(); ++i) {
                double w = C_[l].weights[i];
                size_t p = to_partition(w);
                double w_max = w_max_of(p);
                C_[l].L.push_back(C_[l].P[p].size());
                C_[l].P[p].emplace_back(i, w / w_max);
                C_[l-1].weights[p] += w;
            }
        }
    }

    template <typename Generator>
    size_t sample(Generator&& gen) {
        // sample partition in top layer via linear search
        double x = W_ * real_dist_(gen);
        size_t p = o_;
        size_t p_max = m_ - 1;
        while (x > 0) {
            if (p < p_max) p++; else p = 0;
            x -= C_[0].weights[p];
        }
        // sample index from cascade via rejection sampling
        size_t l = 1;
        while (l <= K) {
            std::uniform_int_distribution<size_t> index_dist(0, C_[l].P[p].size() - 1);
            while (true) {
                auto [i, p_acc] = C_[l].P[p][index_dist(gen)];
                if (real_dist_(gen) < p_acc) {
                    p = i;
                    l++;
                    break;
                }
            }
        }
        return p;
    }

    void update(size_t i, double w_new) {
        double w = C_[K].weights[i];
        double delta = w_new - w;
        W_ += delta;
        update_(K, i, delta);
    }

    size_t push(double w, size_t l = K) {
        size_t i = C_[l].weights.size();
        size_t p = to_partition(0);
        double w_max = w_max_of(p);
        C_[l].weights.push_back(0);
        C_[l].L.push_back(C_[l].P[p].size());
        C_[l].P[p].emplace_back(i, w / w_max);
        update(i, w);
        return i;
    }

    void pop() {
        assert(C_[K].weights.size() > 0);
        size_t i = C_[K].weights.size() - 1;
        update(i, 0);
        size_t p = to_partition(0);
        C_[K].P[p][C_[K].L[i]] = C_[K].P[p].back();
        C_[K].L[C_[K].P[p].back().first] = C_[K].L[i];
        C_[K].P[p].pop_back();
        C_[K].L.pop_back();
        C_[K].weights.pop_back();
    }

private:
    size_t to_partition(double w) {
        if (w > 1) {
            size_t p = std::ceil(std::log2(w));
            return o_ + p;
        } else if (w > 0) {
            size_t p = std::floor(-std::log2(w));
            return o_ <= p ? 0 : o_ - p;
        } else {
            return 0;
        }
    }

    double w_max_of(size_t p) {
        return std::pow(2, p) / std::pow(2, o_);
    }

    void update_(size_t l, size_t i, double delta) {
        double w = C_[l].weights[i];
        double w_new = w + delta;
        C_[l].weights[i] = w_new;
        if (l == 0) return;
        size_t p = to_partition(w);
        size_t p_new = to_partition(w_new);
        assert(p_new < m_);
        double w_max = w_max_of(p_new);
        if (p == p_new) {
            C_[l].P[p][C_[l].L[i]].second = w_new / w_max;
            update_(l - 1, p, delta);
        } else {
            C_[l].P[p][C_[l].L[i]] = C_[l].P[p].back();
            C_[l].L[C_[l].P[p].back().first] = C_[l].L[i];
            C_[l].P[p].pop_back();
            C_[l].L[i] = C_[l].P[p_new].size();
            C_[l].P[p_new].emplace_back(i, w_new / w_max);
            update_(l - 1, p, -w);
            update_(l - 1, p_new, w_new);
        }
    }

    struct Layer {
        std::vector<std::vector<std::pair<size_t, double>>> P;
        std::vector<size_t> L;
        std::vector<double> weights;
    };
    std::array<Layer, K + 1> C_;
    std::uniform_real_distribution<double> real_dist_;
    size_t m_;
    size_t o_;
    double W_;
};

}