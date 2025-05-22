#include "benchmark_utils/algorithm_benchmark.h"
#include "benchmark_utils/random_generator.h"

#include <cstdint>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>

namespace {
constexpr int32_t MATRICES_PER_ITERATION = 10;
constexpr int32_t MIN_SIZE = 1;
constexpr int32_t MAX_SIZE = 300;
constexpr int32_t SIZE_STEP = 1;
}// namespace

using namespace LinAlgTools::Tests;
using namespace LinAlgTools::Tests::Utils;

RandomGenerator<std::complex<long double>> generator(20);

void RunQRPerformanceTest(QRMethod method, const std::string& methodName) {
        std::string filename = "qr_performance_" + methodName + ".csv";
        std::ofstream outFile(filename, std::ios::app);

        std::cout << "\nQR Decomposition Performance Test (" << methodName << ")\n";
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
                auto stats = GetQRTimingStatistics(size, MATRICES_PER_ITERATION, method, generator);

                std::cout << std::setw(10) << size
                          << std::setw(15) << stats.mean
                          << std::setw(15) << stats.min
                          << std::setw(15) << stats.max
                          << std::setw(15) << std::fixed << std::setprecision(2)
                          << stats.stddev << '\n';

                outFile << methodName << ","
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

TEST(TEST_PERFORMANCE_QR, HouseholderPerformance) {
        RunQRPerformanceTest(QRMethod::Householder, "Householder");
}

TEST(TEST_PERFORMANCE_QR, GivensPerformance) {
        RunQRPerformanceTest(QRMethod::Givens, "Givens");
}

