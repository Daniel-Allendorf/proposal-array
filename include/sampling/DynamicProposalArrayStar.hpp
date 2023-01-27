#pragma once
#include <cassert>
#include <queue>
#include <random>
#include <vector>

namespace sampling {

using index_t = uint64_t;

class DynamicProposalArrayStar {
public:
    DynamicProposalArrayStar(const std::vector<double>& weights, double W) :
        weights_(weights), counts_(weights.size()), N_(weights.size()), W_(W),
        avg_(W / weights.size()), p_(0), q_(weights_.size() - 1) {
        assert(N_ > 0);
        P_.reserve(3 * N_);
        B_.reserve(3 * N_);
        construct();
    }

    DynamicProposalArrayStar() : N_(0) {}

    template <typename Generator>
    index_t sample(Generator&& gen) {
        assert(N_ > 0);
        std::uniform_int_distribution<size_t> loc_dist(0, P_.size() - 1);
        std::uniform_real_distribution<double> real_dist(0, 1);
        do {
            index_t element = P_[loc_dist(gen)];
            double p_acc = (weights_[element] / counts_[element]) / (2 * W_ / N_);
            if (real_dist(gen) < p_acc) {
                return element;
            }
        } while (true);
    }

    void update(index_t i, double w) {
        assert(i <= N_);
        W_ += w - weights_[i];
        weights_[i] = w;

        index_t new_count = std::ceil(w / (W_ / N_));
        index_t old_count = counts_[i];
        if (new_count > old_count) {
            for (index_t c = old_count; c < new_count; ++c) {
                insert(i);
            }
        } else if (new_count < old_count) {
            for (index_t c = new_count; c < old_count; ++c) {
                erase(i);
            }
        }
        counts_[i] = new_count;

        int64_t s = 3 * N_ * std::log2((W_ / N_) / avg_);
        if (s > 0) s++;
        if (s < 0) s--;
        while (s > 0) {
            index_t new_count_o = std::ceil(weights_[p_] / (W_ / N_));
            index_t old_count_o = counts_[p_];
            if (new_count_o < old_count_o) {
                erase(p_);
                counts_[p_]--;
            } else {
                p_++;
                if (p_ == N_) p_ = 0;
            }
            s--;
        }
        while (s < 0) {
            index_t new_count_o = std::ceil(weights_[q_] / (W_ / N_));
            index_t old_count_o = counts_[q_];
            if (new_count_o > old_count_o) {
                insert(q_);
                counts_[q_]++;
            } else {
                if (q_ == 0) q_ = N_;
                q_--;
            }
            s++;
        }
        avg_ = W_ / N_;
    }

    std::string name() {
        return "ProposalArrayStar";
    }

private:
    void construct() {
        P_.clear();
        B_.clear();
        L_ = std::vector<std::vector<index_t>>(N_);
        for (size_t i = 0; i < N_; ++i) {
            counts_[i] = std::ceil(weights_[i] / (W_ / N_));
            for (size_t j = 0; j < counts_[i]; ++j) {
                insert(i);
            }
        }
    }

    void insert(index_t i) {
        B_.push_back(L_[i].size());
        L_[i].push_back(P_.size());
        P_.push_back(i);
    }

    void erase(index_t i) {
        P_[L_[i].back()] = P_.back();
        B_[L_[i].back()] = B_.back();
        L_[P_.back()][B_.back()] = L_[i].back();
        P_.pop_back();
        B_.pop_back();
        L_[i].pop_back();
        assert(L_[i].size() > 0);
    }

    std::vector<index_t> P_;
    std::vector<std::vector<index_t>> L_;
    std::vector<index_t> B_;
    std::vector<index_t> counts_;
    std::vector<double> weights_;
    size_t N_;
    double W_;
    double avg_;
    index_t p_;
    index_t q_;
};

}