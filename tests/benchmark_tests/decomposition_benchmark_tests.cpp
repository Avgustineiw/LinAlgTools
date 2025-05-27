#include "benchmark_utils/algorithm_benchmark.h"
#include "../../src/core/random_generator.h"

#include <gtest/gtest.h>

using namespace LinAlgTools::Tests::Utils;
using namespace LinAlgTools::Core;

TEST(TEST_PERFORMANCE_QR, HouseholderQRPerformance) {
        RunPerformanceTest(Algorithm::HouseholderQR, 1, 50);
}

TEST(TEST_PERFORMANCE_QR, GivensQRPerformance) {
        RunPerformanceTest(Algorithm::GivensQR, 1, 50);
}

TEST(TEST_PERFORMANCE_SVD, RealSchurPerformance) {
        RunPerformanceTest(Algorithm::RealSchur, 1, 50, 1, 10, RandomGenerator<long double>(20));
}

TEST(TEST_PERFORMANCE_SVD, NaiveSVDPerformance) {
        RunPerformanceTest(Algorithm::NaiveSVD, 1, 25);
}

