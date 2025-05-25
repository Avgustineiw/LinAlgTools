#include "benchmark_utils/algorithm_benchmark.h"
#include "benchmark_utils/random_generator.h"

#include <gtest/gtest.h>

using namespace LinAlgTools::Tests::Utils;

TEST(TEST_PERFORMANCE_QR, HouseholderQRPerformance) {
        RunPerformanceTest(Method::HouseholderQR, 1, 50);
}

TEST(TEST_PERFORMANCE_QR, GivensQRPerformance) {
        RunPerformanceTest(Method::GivensQR, 1, 50);
}

TEST(TEST_PERFORMANCE_SVD, RealSchurPerformance) {
        RunPerformanceTest(Method::RealSchur, 1, 20, 1, 10, RandomGenerator<long double>(20));
}

TEST(TEST_PERFORMANCE_SVD, NaiveSVDPerformance) {
        RunPerformanceTest(Method::NaiveSVD, 1, 25);
}

