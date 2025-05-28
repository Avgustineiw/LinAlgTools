#pragma once

#include "../../../src/algorithms/qr_decomposition.h"
#include "../../../src/algorithms/schur_decomposition.h"
#include "../../../src/algorithms/svd.h"
#include "../../../src/core/random_generator.h"

#include <cassert>
#include <chrono>
#include <complex>
#include <cstdint>
#include <fstream>
#include <gtest/gtest.h>

namespace LinAlgTools::Tests::Utils {
enum class AlgorithmName
{
        HouseholderQR,
        GivensQR,
        RealSchur,
        NaiveSVD,
};

namespace Implementation {
struct TimingResult
{
        double mean = 0;
        double min = 0;
        double max = 0;
        double stddev = 0.0;
        size_t count = 0;
};

using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::milliseconds;
using namespace LinAlgTools::Tests;
using namespace LinAlgTools::Tests::Utils;

inline TimingResult CalculateStatistics(const std::vector<double>& data) {
        TimingResult result;
        if (data.empty()) {
                return result;
        }

        result.count = data.size();

        result.min = data[0];
        result.max = data[0];
        double sum = 0;

        for (const auto t: data) {
                sum += t;
                if (t < result.min) result.min = t;
                if (t > result.max) result.max = t;
        }

        result.mean = sum / static_cast<double>(result.count);

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
TimingResult GetTimingStatistics(AlgorithmName algorithm,
                                 Core::RandomGenerator<T> generator,
                                 int32_t size, int32_t iterations) {
        std::vector<double> data;

        for (int32_t i = 0; i < iterations; i++) {
                auto matrix = generator.GetRandomDenseMatrix(size, size, 1e-10, 1e+10);

                Clock::time_point start;
                Clock::time_point end;
                switch (algorithm) {
                        case AlgorithmName::HouseholderQR: {
                                start = Clock::now();
                                auto result = LinAlgTools::Algorithm::HouseholderQR(matrix);
                                end = Clock::now();
                                break;
                        }
                        case AlgorithmName::GivensQR: {
                                start = Clock::now();
                                auto result = LinAlgTools::Algorithm::GivensQR(matrix);
                                end = Clock::now();
                                break;
                        }
                        case AlgorithmName::RealSchur: {
                                start = Clock::now();
                                auto result = LinAlgTools::Algorithm::RealSchur(matrix);
                                end = Clock::now();
                                break;
                        }

                        case AlgorithmName::NaiveSVD: {
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

std::string GetNameOfAlgorithm(AlgorithmName algorithm) {
        switch (algorithm) {
                case Implementation::AlgorithmName::HouseholderQR:
                        return "HouseholderQR";
                        break;
                case Implementation::AlgorithmName::GivensQR:
                        return "GivensQR";
                        break;
                case Implementation::AlgorithmName::RealSchur:
                        return "RealSchur";
                        break;
                case Implementation::AlgorithmName::NaiveSVD:
                        return "NaiveSVD";
                        break;
                default:
                        assert("Unknown algorithm");
        }
}
}//namespace Implementation

template<typename T = std::complex<long double>>
void RunPerformanceTest(Implementation::AlgorithmName algorithm,
                        int32_t min_size = 1, int32_t max_size = 100,
                        int32_t size_step = 1, int32_t matrices_per_iteration = 10,
                        Core::RandomGenerator<T> generator = Core::RandomGenerator<T>(20)) {
        std::string algorithm_name = Implementation::GetNameOfAlgorithm(algorithm);

        std::string filename = algorithm_name + "_performance.csv";
        std::ofstream outFile(filename, std::ios::app);

        std::cout << "\nPerformance Test (" << algorithm_name << ")\n";
        std::cout << "---------------------------------\n";
        std::cout << std::setw(10) << "Size"
                  << std::setw(15) << "Mean (ms)"
                  << std::setw(15) << "Min (ms)"
                  << std::setw(15) << "Max (ms)"
                  << std::setw(15) << "StdDev\n";

        outFile << "Algorithm,Size,Mean(ms),Min(ms),Max (ms),StdDev\n";

        for (int32_t size = min_size; size <= max_size; size += size_step) {
                auto stats = Implementation::GetTimingStatistics(algorithm, generator, size, matrices_per_iteration);

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

        std::cout << "\nResults saved to: " << filename << '\n';
}
}//namespace LinAlgTools::Tests::Utils

