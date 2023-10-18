#pragma once
#ifndef SCOPED_TIMER_HPP
#define SCOPED_TIMER_HPP

#include <iostream>
#include <string>
#include <sstream>
#include <chrono>

namespace tools {

class ScopedTimer {
    using Clock = std::chrono::high_resolution_clock;

    std::string _prefix;
    Clock::time_point _begin;

    uint64_t _scale;
    double _offset;

    double *_output;

public:
    ScopedTimer() : _begin(Clock::now()), _scale(0), _offset(0), _output(nullptr) {}

    ScopedTimer(const std::string &prefix, uint64_t scale = 0, double offset = 0.0) : _prefix(prefix), _begin(Clock::now()),
                                                                                      _scale(scale), _offset(offset), _output(nullptr) {}

    ScopedTimer(double &output) : _begin(Clock::now()), _scale(1), _offset(0), _output(&output) {}

    ~ScopedTimer() {
        if (!_prefix.empty())
            report();

        if (_output)
            *_output = elapsedSeconds();
    }

    void start() {
        _begin = Clock::now();
    }

    double elapsedSeconds() const {
        const auto t2 = Clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - _begin);

        return time_span.count() - _offset;
    }

    double report() const {
        return report(_prefix);
    }

    double report(const std::string &prefix) const {
        const double timeMS = elapsedSeconds() * 1e3;

        if (!_scale) {
            std::cout << prefix << " Time elapsed: " << timeMS << "ms" << std::endl;
        } else {
            std::cout << prefix << " Time elapsed: " << timeMS << "ms / " << _scale << " = " << (1e3 * timeMS / _scale) << "us"
                      << std::endl;
        }

        return timeMS;
    }
};

}

#endif
