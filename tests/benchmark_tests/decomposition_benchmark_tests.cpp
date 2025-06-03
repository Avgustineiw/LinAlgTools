#include "benchmark_utils/algorithm_benchmark.h"

#include <algorithm>
#include <gtest/gtest.h>

using namespace LinAlgTools::Tests::Utils;
using namespace LinAlgTools::Core;

namespace BenchmarkConfig {
inline constexpr int32_t min_iterations = 1;
inline constexpr int32_t max_iterations = 100;
}

TEST(TEST_PERFORMANCE_QR, HouseholderQRPerformance) {
        RunPerformanceTest(AlgorithmId::HouseholderQR, 
                           BenchmarkConfig::min_iterations, 
                           BenchmarkConfig::max_iterations);
}

TEST(TEST_PERFORMANCE_QR, GivensQRPerformance) {
        RunPerformanceTest(AlgorithmId::GivensQR, 
                           BenchmarkConfig::min_iterations,
                           BenchmarkConfig::max_iterations);
}

TEST(TEST_PERFORMANCE_SVD, RealSchurPerformance) {
        RunPerformanceTest<double>(AlgorithmId::RealSchur, 
                                   BenchmarkConfig::min_iterations, 
                                   BenchmarkConfig::max_iterations);
}

TEST(TEST_PERFORMANCE_SVD, NaiveSVDPerformance) {
        RunPerformanceTest(AlgorithmId::NaiveSVD, 
                           BenchmarkConfig::min_iterations, 
                           BenchmarkConfig::max_iterations);
}

