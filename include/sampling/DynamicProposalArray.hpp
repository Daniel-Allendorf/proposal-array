#pragma once
#include <cassert>
#include <random>
#include <vector>

namespace sampling {

using index_t = uint64_t;

class DynamicProposalArray {
public:
    DynamicProposalArray(const std::vector<double>& weights, double W) :
        weights_(weights), counts_(weights.size()), N_(weights.size()), W_(W), avg_(W / weights.size()) {
        assert(N_ > 0);
        P_.reserve(3 * N_);
        B_.reserve(3 * N_);
        construct();
    }

    DynamicProposalArray() : N_(0) {}

    template <typename Generator>
    index_t sample(Generator&& gen) {
        assert(N_ > 0);
        std::uniform_int_distribution<size_t> loc_dist(0, P_.size() - 1);
        std::uniform_real_distribution<double> real_dist(0, 1);
        do {
            index_t element = P_[loc_dist(gen)];
            assert(weights_[element] / counts_[element] <= 2 * avg_);
            double p_acc = (weights_[element] / counts_[element]) / (2 * avg_);
            if (real_dist(gen) < p_acc) {
                return element;
            }
        } while (true);
    }

    void update(index_t i, double w) {
        assert(i <= N_);
        index_t new_count;
        index_t old_count;
        if (i == N_) { // insertion
            assert(w > 0);
            new_count = std::ceil(w / avg_);
            old_count = 0;
            N_++;
            W_ += w;
            weights_.push_back(w);
            counts_.push_back(new_count);
        } else if (w == 0.) { // deletion
            assert(i == N_ - 1);
            new_count = 0;
            old_count = counts_[i];
            N_--;
            W_ -= weights_[i];
            weights_.pop_back();
            counts_.pop_back();
        } else {
            new_count = std::ceil(w / avg_);
            old_count = counts_[i];
            W_ += w - weights_[i];
            weights_[i] = w;
            counts_[i] = new_count;
        }

        double new_avg = W_ / N_;
        if (new_avg < avg_ / 2 || new_avg > 2 * avg_) {
            avg_ = new_avg;
            construct();
        } else {
            if (new_count > old_count) {
                for (index_t c = old_count; c < new_count; ++c) {
                    insert(i);
                }
            } else if (new_count < old_count) {
                for (index_t c = new_count; c < old_count; ++c) {
                    erase(i);
                }
            }
        }
    }

    std::string name() {
        return "ProposalArray";
    }

private:
    void construct() {
        P_.clear();
        B_.clear();
        L_ = std::vector<std::vector<index_t>>(N_);
        for (size_t i = 0; i < N_; ++i) {
            counts_[i] = std::ceil(weights_[i] / avg_);
            for (size_t j = 0; j < counts_[i]; ++j) {
                insert(i);
            }
        }
    }

    void insert(index_t i) {
        if (L_.size() <= i) {
            assert(i == N_ - 1);
            L_.push_back(std::vector<index_t>());
        }
        B_.push_back(L_[i].size());
        L_[i].push_back(P_.size());
        P_.push_back(i);
    }

    void erase(index_t i) {
        assert(L_[i].size() > 0);
        P_[L_[i].back()] = P_.back();
        B_[L_[i].back()] = B_.back();
        L_[P_.back()][B_.back()] = L_[i].back();
        P_.pop_back();
        B_.pop_back();
        L_[i].pop_back();
        if (L_[i].empty()) {
            assert(i == N_);
            L_.pop_back();
        }
    }

    std::vector<index_t> P_;
    std::vector<std::vector<index_t>> L_;
    std::vector<index_t> B_;
    std::vector<index_t> counts_;
    std::vector<double> weights_;
    size_t N_;
    double W_;
    double avg_;
};

}