#pragma once

#include "../../../src/algorithms/qr_decomposition.h"
#include "../../../src/algorithms/svd.h"
#include "../../../src/algorithms/schur_decomposition.h"
#include "random_generator.h"

#include <chrono>
#include <complex>
#include <cstdint>
#include <gtest/gtest.h>

namespace LinAlgTools::Tests::Utils {
enum class Method
{
        HouseholderQR,
        GivensQR,
        RealSchur,
        NaiveSVD,
};

struct TimingResult
{
        int64_t mean = 0;
        int64_t min = 0;
        int64_t max = 0;
        double stddev = 0.0;
        size_t count = 0;
};

using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::milliseconds;

TimingResult CalculateStatistics(const std::vector<int64_t>& data) {
        TimingResult result;
        if (data.empty()) {
                return result;
        }

        result.count = data.size();

        result.min = data[0];
        result.max = data[0];
        int64_t sum = 0;

        for (const auto t: data) {
                sum += t;
                if (t < result.min) result.min = t;
                if (t > result.max) result.max = t;
        }

        result.mean = sum / static_cast<int64_t>(result.count);

        if (result.count > 1) {
                double variance = 0.0;
                for (const auto t: data) {
                        variance += std::pow(t - result.mean, 2);
                }
                result.stddev = std::sqrt(variance / result.count);
        }

        return result;
}

TimingResult GetTimingStatistics(Method method,
                                 RandomGenerator<std::complex<long double>> generator,
                                 size_t size, size_t iterations) {
        std::vector<int64_t> data;

        for (int i = 0; i < iterations; i++) {
                auto matrix = generator.GetRandomSparseMatrix(size);

                Clock::time_point start;
                Clock::time_point end;
                switch (method) {
                        case Method::HouseholderQR: {
                                start = Clock::now();
                                auto result = LinAlgTools::Algorithm::HouseholderQR(matrix);
                                end = Clock::now();
                                break;
                        }
                        case Method::GivensQR: {
                                start = Clock::now();
                                auto result = LinAlgTools::Algorithm::GivensQR(matrix);
                                end = Clock::now();
                                break;
                        }
                        case Method::RealSchur: {
                                start = Clock::now();
                                auto result = LinAlgTools::Algorithm::RealSchur(matrix);
                                end = Clock::now();
                                break;
                        }

                        case Method::NaiveSVD: {
                                start = Clock::now();
                                auto result = LinAlgTools::Algorithm::NaiveSVD(matrix);
                                end = Clock::now();
                                break;
                        }
                        default: {
                                std::cout << "Invalid QR Algorithm";
                                break;
                        }
                }

                data.push_back(std::chrono::duration_cast<Ms>(end - start).count());
        }

        return CalculateStatistics(data);
}
}//namespace LinAlgTools::Tests::Utils

