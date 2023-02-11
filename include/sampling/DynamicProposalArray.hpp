#pragma once
#include <cassert>
#include <random>
#include <vector>

namespace sampling {

class DynamicProposalArray {
public:
    DynamicProposalArray(const std::vector<double>& weights) :
        weights_(weights), R_(weights.size()), real_dist_(0, 1), W_(0.0) {
        assert(weights.size() > 0);
        N_ = weights_.size();
        for (auto w : weights) W_ += w;
        avg_ = W_ / N_;
        P_.reserve(2 * N_);
        construct();
    }

    template <typename Generator>
    size_t sample(Generator&& gen) {
        std::uniform_int_distribution<size_t> entry_dist(0, R_.size() + P_.size() - 1);
        do {
            auto i = entry_dist(gen);
            if (i < R_.size()) {
                double p_acc = R_[i];
                if (real_dist_(gen) < p_acc) {
                    return i;
                }
            } else {
                return P_[i - R_.size()].first;
            }
        } while (true);
    }

    void update(size_t i, double w) {
        assert(i <= N_);

        double w_old;
        if (i == N_) { // insertion
            assert(w > 0);
            w_old = 0;
            N_++;
            W_ += w;
            weights_.push_back(w);
            R_.push_back(0.0);
        } else if (w == 0.) { // deletion
            assert(i == N_ - 1);
            w_old = weights_[i];
            N_--;
            W_ -= w_old;
            weights_.pop_back();
            R_.pop_back();
        } else {
            w_old = weights_[i];
            W_ += w - w_old;
            weights_[i] = w;
        }

        double new_avg = W_ / N_;
        if (new_avg < avg_ / 2 || new_avg > 2 * avg_) {
            avg_ = new_avg;
            reconstruct();
        } else {
            if (w > w_old) {
                size_t count = std::floor(w / avg_);
                for (size_t c = std::floor(w_old / avg_); c < count; ++c) {
                    insert(i);
                }
                R_[i] = (w / avg_) - count;
            } else if (w < w_old) {
                size_t count = std::floor(w / avg_);
                for (size_t c = std::floor(w_old / avg_); c > count; --c) {
                    erase(i);
                }
                R_[i] = (w / avg_) - count;
            }
        }
    }

private:
    void construct() {
        P_.clear();
        L_ = std::vector<std::vector<size_t>>(N_);
        for (size_t i = 0; i < N_; ++i) {
            double weight = weights_[i];
            size_t count = std::floor(weight / avg_);
            for (size_t j = 0; j < count; ++j) {
                insert(i);
            }
            R_[i] = (weight / avg_) - count;
        }
    }

    void reconstruct() {
        std::vector<size_t> counts(N_, 0);
        for (auto [i, b] : P_) counts[i]++;
        for (size_t i = 0; i < N_; ++i) {
            double weight = weights_[i];
            size_t count = std::floor(weight / avg_);
            for (size_t c = counts[i]; c < count; ++c) {
                insert(i);
            }
            for (size_t c = counts[i]; c > count; --c) {
                erase(i);
            }
            R_[i] = (weight / avg_) - count;
        }
    }

    void insert(size_t i) {
        if (L_.size() <= i) {
            assert(i == N_ - 1);
            L_.push_back(std::vector<size_t>());
        }
        L_[i].push_back(P_.size());
        P_.emplace_back(i, L_[i].size() - 1);
    }

    void erase(size_t i) {
        assert(L_[i].size() > 0);
        P_[L_[i].back()] = P_.back();
        L_[P_.back().first][P_.back().second] = L_[i].back();
        P_.pop_back();
        L_[i].pop_back();
        if (L_[i].empty() && i == weights_.size()) {
            assert(i == N_);
            L_.pop_back();
        }
    }

    std::vector<double> weights_;
    std::vector<double> R_;
    std::vector<std::pair<size_t, size_t>> P_;
    std::vector<std::vector<size_t>> L_;
    std::uniform_real_distribution<double> real_dist_;
    size_t N_;
    double W_;
    double avg_;
};

}