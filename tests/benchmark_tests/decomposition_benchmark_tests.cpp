#include "benchmark_utils/algorithm_benchmark.h"

#include <gtest/gtest.h>

using namespace LinAlgTools::Tests::Utils;
using namespace LinAlgTools::Core;

TEST(TEST_PERFORMANCE_QR, HouseholderQRPerformance) {
        RunPerformanceTest(AlgorithmName::HouseholderQR, 1, 100);
}

TEST(TEST_PERFORMANCE_QR, GivensQRPerformance) {
        RunPerformanceTest(AlgorithmName::GivensQR, 1, 100);
}

TEST(TEST_PERFORMANCE_SVD, RealSchurPerformance) {
        RunPerformanceTest<double>(AlgorithmName::RealSchur, 1, 50);
}

TEST(TEST_PERFORMANCE_SVD, NaiveSVDPerformance) {
        RunPerformanceTest(AlgorithmName::NaiveSVD, 1, 50);
}

