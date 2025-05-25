#include "benchmark_utils/algorithm_benchmark.h"
#include "benchmark_utils/random_generator.h"

#include <cstdint>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>

namespace {
constexpr int32_t MATRICES_PER_ITERATION = 10;
constexpr int32_t MIN_SIZE = 1;
constexpr int32_t MAX_SIZE = 100;
constexpr int32_t SIZE_STEP = 1;
}// namespace

using namespace LinAlgTools::Tests;
using namespace LinAlgTools::Tests::Utils;

void RunPerformanceTest(Method method,
                        RandomGenerator<std::complex<long double>> generator = RandomGenerator<std::complex<long double>>(20)) {
        std::string method_name;
        switch (method) {
                case Method::HouseholderQR:
                        method_name = "HouseholderQR";
                        break;
                case Method::GivensQR:
                        method_name = "GivensQR";
                        break;
                case Method::RealSchur:
                        method_name = "RealSchur";
                        break;
                case Method::NaiveSVD:
                        method_name = "NaiveSVD";
                        break;
                default:
                        throw std::runtime_error("Unknown method");
        }

        std::string filename = method_name + "_performance.csv";
        std::ofstream outFile(filename, std::ios::app);

        std::cout << "\nPerformance Test (" << method_name << ")\n";
        std::cout << "---------------------------------\n";
        std::cout << std::setw(10) << "Size"
                  << std::setw(15) << "Mean (ms)"
                  << std::setw(15) << "Min (ms)"
                  << std::setw(15) << "Max (ms)"
                  << std::setw(15) << "StdDev\n";

        if (outFile.tellp() == 0) {
                outFile << "Algorithm,Size,Mean (ms),Min (ms),Max (ms),StdDev\n";
        }

        for (int32_t size = MIN_SIZE; size <= MAX_SIZE; size += SIZE_STEP) {
                auto stats = GetTimingStatistics(method, generator, size, MATRICES_PER_ITERATION);

                std::cout << std::setw(10) << size
                          << std::setw(15) << stats.mean
                          << std::setw(15) << stats.min
                          << std::setw(15) << stats.max
                          << std::setw(15) << std::fixed << std::setprecision(2)
                          << stats.stddev << '\n';

                outFile << method_name << ","
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

TEST(TEST_PERFORMANCE_QR, HouseholderQRPerformance) {
        RunPerformanceTest(Method::HouseholderQR);
}

TEST(TEST_PERFORMANCE_QR, GivensQRPerformance) {
        RunPerformanceTest(Method::GivensQR);
}

TEST(TEST_PERFORMANCE_SVD, NaiveSVDPerformance) {
        RunPerformanceTest(Method::NaiveSVD);
}

