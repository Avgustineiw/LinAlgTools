#pragma once

#include "../../../src/algorithms/qr_decomposition.h"
#include "../../../src/algorithms/schur_decomposition.h"
#include "../../../src/algorithms/svd.h"
#include "../../../src/core/random_generator.h"

#include <chrono>
#include <complex>
#include <cstdint>
#include <fstream>
#include <gtest/gtest.h>

namespace LinAlgTools::Tests::Utils {
namespace {
constexpr int32_t MIN_SIZE = 1;
constexpr int32_t MAX_SIZE = 100;
constexpr int32_t SIZE_STEP = 1;
constexpr int32_t MATRICES_PER_ITERATION = 10;
}// namespace

enum class Algorithm 
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
using namespace LinAlgTools::Tests;
using namespace LinAlgTools::Tests::Utils;

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

template<typename T = std::complex<long double>>
TimingResult GetTimingStatistics(Algorithm algorithm,
                                 Core::RandomGenerator<T> generator,
                                 int32_t size, int32_t iterations) {
        std::vector<int64_t> data;

        for (int i = 0; i < iterations; i++) {
                auto matrix = generator.GetRandomSparseMatrix(size);

                Clock::time_point start;
                Clock::time_point end;
                switch (algorithm) {
                        case Algorithm::HouseholderQR: {
                                start = Clock::now();
                                auto result = LinAlgTools::Algorithm::HouseholderQR(matrix);
                                end = Clock::now();
                                break;
                        }
                        case Algorithm::GivensQR: {
                                start = Clock::now();
                                auto result = LinAlgTools::Algorithm::GivensQR(matrix);
                                end = Clock::now();
                                break;
                        }
                        case Algorithm::RealSchur: {
                                start = Clock::now();
                                auto result = LinAlgTools::Algorithm::RealSchur(matrix);
                                end = Clock::now();
                                break;
                        }

                        case Algorithm::NaiveSVD: {
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

template<typename T = std::complex<long double>>
void RunPerformanceTest(Algorithm algorithm,
                        int32_t min_size = MIN_SIZE, int32_t max_size = MAX_SIZE, 
                        int32_t size_step = SIZE_STEP, int32_t matrices_per_iteration = MATRICES_PER_ITERATION,
                        Core::RandomGenerator<T> generator = Core::RandomGenerator<T>(20)) {
        std::string algorithm_name;
        switch (algorithm) {
                case Algorithm::HouseholderQR:
                        algorithm_name = "HouseholderQR";
                        break;
                case Algorithm::GivensQR:
                        algorithm_name = "GivensQR";
                        break;
                case Algorithm::RealSchur:
                        algorithm_name = "RealSchur";
                        break;
                case Algorithm::NaiveSVD:
                        algorithm_name = "NaiveSVD";
                        break;
                default:
                        throw std::runtime_error("Unknown method");
        }

        std::string filename = algorithm_name + "_performance.csv";
        std::ofstream outFile(filename, std::ios::app);

        std::cout << "\nPerformance Test (" << algorithm_name << ")\n";
        std::cout << "---------------------------------\n";
        std::cout << std::setw(10) << "Size"
                  << std::setw(15) << "Mean (ms)"
                  << std::setw(15) << "Min (ms)"
                  << std::setw(15) << "Max (ms)"
                  << std::setw(15) << "StdDev\n";

        if (outFile.tellp() == 0) {
                outFile << "Algorithm,Size,Mean (ms),Min (ms),Max (ms),StdDev\n";
        }

        for (int32_t size = min_size; size <= max_size; size += size_step) {
                auto stats = GetTimingStatistics(algorithm, generator, size, matrices_per_iteration);

                std::cout << std::setw(10) << size
                          << std::setw(15) << stats.mean
                          << std::setw(15) << stats.min
                          << std::setw(15) << stats.max
                          << std::setw(15) << std::fixed << std::setprecision(2)
                          << stats.stddev << '\n';

                outFile << algorithm_name << ","
                        << size << ","
                        << stats.mean << ","
                        << stats.min << ","
                        << stats.max << ","
                        << std::fixed << std::setprecision(6)
                        << stats.stddev << '\n';
        }

        outFile.close();
        std::cout << "\nResults saved to: " << filename << '\n';
}
}//namespace LinAlgTools::Tests::Utils

